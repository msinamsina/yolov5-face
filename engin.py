import argparse
import logging
import math
import os
import random
import time
from pathlib import Path
from threading import Thread
from warnings import warn

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test  # import test.py to get mAP after each epoch
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.face_datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    print_mutation, set_logging
from utils.google_utils import attempt_download
from utils.loss import compute_loss
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first

logger = logging.getLogger(__name__)

try:
    import wandb
except ImportError:
    wandb = None
    logger.info("Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)")


class Trainer:
    def __init__(self, hyp, opt, device, tb_writer=None, wandb=None):
        self.hyp = hyp
        self.opt = opt
        self.device = device
        self.tb_writer = tb_writer
        self.wandb = wandb

        self.save_dir = Path(opt.save_dir)
        self.epochs = opt.epochs
        self.batch_size = opt.batch_size
        self.total_batch_size = opt.total_batch_size
        self.weights = opt.weights
        self.rank = opt.global_rank

        self.wdir = self.save_dir / 'weights'
        self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
        self.last = self.wdir / 'last.pt'
        self.best = self.wdir / 'best.pt'
        self.results_file = self.save_dir / 'results.txt'

        self.cuda = device.type != 'cpu'
        init_seeds(2 + self.rank)

        # Save run settings
        with open(self.save_dir / 'hyp.yaml', 'w') as f:
            yaml.dump(hyp, f, sort_keys=False)
        with open(self.save_dir / 'opt.yaml', 'w') as f:
            yaml.dump(vars(opt), f, sort_keys=False)

        # data
        with open(opt.data) as f:
            self.data_dict = yaml.load(f, Loader=yaml.FullLoader)
        with torch_distributed_zero_first(self.rank):
            check_dataset(self.data_dict)

        self.train_path = self.data_dict['train']
        self.test_path = self.data_dict['val']
        self.nc = 1 if opt.single_cls else int(self.data_dict['nc'])  # number of classes

        # class names
        self.names = ['item'] if opt.single_cls and len(self.data_dict['names']) != 1 else self.data_dict['names']

        # check number of classes is equal to nc
        assert len(self.names) == self.nc, f'{len(self.names)} names found for nc={self.nc} dataset in {opt.data}'

        # creat MODEL and load weights (download if pretrained weights not exist)
        self.pretrained = self.weights.endswith('.pt')
        if self.pretrained:
            with torch_distributed_zero_first(self.rank):
                # download if not found locally
                attempt_download(self.weights)
            # load checkpoint
            self.ckpt = torch.load(self.weights, map_location=self.device)
            # force autoanchor
            if self.hyp.get('anchors'):
                self.ckpt['model'].yaml['anchors'] = round(self.hyp['anchors'])

            # creat model and load weights
            self.model = Model(self.opt.cfg or self.ckpt['model'].yaml, ch=3, nc=self.nc).to(self.device)
            # exclude keys
            exclude = ['anchor'] if self.opt.cfg or self.hyp.get('anchors') else []
            # to Float32
            state_dict = self.ckpt['model'].float().state_dict()
            state_dict = intersect_dicts(state_dict, self.model.state_dict(), exclude=exclude)  # intersect
            # load state_dict
            self.model.load_state_dict(state_dict, strict=False)
            logger.info(f'Transferred {len(state_dict)}/{len(self.model.state_dict())} items from {self.weights}')
        else:
            self.model = Model(self.opt.cfg, ch=3, nc=self.nc).to(self.device)

        # parameter names to freeze (full or partial)
        freeze = []
        for k, v in self.model.named_parameters():
            v.requires_grad = True
            if any(x in k for x in freeze):
                print('freezing %s' % k)
                v.requires_grad = False

        # nominal batch size
        self.nominal_batch_size = 64
        # accumulate loss before optimizing
        self.accumulate = max(round(self.nominal_batch_size / self.total_batch_size), 1)
        # scale weight_decay
        self.hyp['weight_decay'] *= self.total_batch_size * self.accumulate / self.nominal_batch_size

        # optimizer parameter groups
        self.pg0, self.pg1, self.pg2 = [], [], []
        for k, v in self.model.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                self.pg2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d):
                self.pg0.append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                self.pg1.append(v.weight)

        if self.opt.adam:
            self.optimizer = optim.Adam(self.pg0, lr=self.hyp['lr0'], betas=(self.hyp['momentum'], 0.999))
        else:
            self.optimizer = optim.SGD(self.pg0, lr=self.hyp['lr0'], momentum=self.hyp['momentum'], nesterov=True)

        self.optimizer.add_param_group({'params': self.pg1, 'weight_decay': self.hyp['weight_decay']})
        self.optimizer.add_param_group({'params': self.pg2})
        logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(self.pg2), len(self.pg1), len(self.pg0)))
        del self.pg0, self.pg1, self.pg2

        self.lf = lambda x: ((1 + math.cos(x * math.pi / self.epochs)) / 2) * (1 - self.hyp['lrf']) + self.hyp['lrf']
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

        # Logging
        if self.wandb and self.wandb.run is None:
            self.opt.hyp = self.hyp
            self.wandb_run = self.wandb.init(
                config=self.opt, resume="allow",
                project='YOLOv5' if self.opt.project == 'runs/train' else Path(self.opt.project).stem,
                name=self.save_dir.stem,
                id=self.ckpt.get('wandb_id') if 'ckpt' in locals() else None)

        self.loggers = {'wandb': self.wandb}

        # Resume
        self.start_epoch, self.best_fitness = 0, 0.0
        if self.pretrained:
            # Optimizer
            if self.ckpt['optimizer'] is not None:
                self.optimizer.load_state_dict(self.ckpt['optimizer'])
                self.best_fitness = self.ckpt['best_fitness']

            # Results
            if self.ckpt.get('training_results') is not None:
                with open(self.results_file, 'w') as file:
                    file.write(self.ckpt['training_results'])  # write results.txt

            # Epochs
            if self.opt.resume:
                assert self.start_epoch > 0, \
                    f'{self.weights} training to {self.epochs} epochs is finished, nothing to resume.'
            if self.epochs < self.start_epoch:
                logger.info(f'{self.weights} has been trained for {self.ckpt["epoch"]}'
                            f' epochs. Fine-tuning for {self.epochs} additional epochs.')
                self.epochs += self.ckpt['epoch']
            del self.ckpt, state_dict

        # grid size (max stride)
        self.grid_size = int(max(self.model.stride))
        self.img_size, self.test_img_size = [check_img_size(x, self.grid_size) for x in self.opt.img_size]

        # DP mode
        if self.cuda and self.rank == -1 and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        # SyncBatchNorm
        if self.opt.sync_bn and self.cuda and self.rank != -1:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.device)
            logger.info('Using SyncBatchNorm()')

        # EMA
        self.ema = ModelEMA(self.model) if self.rank in [-1, 0] else None

        # DDP mode
        if self.cuda and self.rank != -1:
            self.model = DDP(self.model, device_ids=[self.opt.local_rank], output_device=self.opt.local_rank)

        # Train loader
        self.dataloader, self.dataset = create_dataloader(self.train_path, self.img_size,
                                                          self.batch_size, self.grid_size,
                                                          self.opt, hyp=self.hyp,
                                                          augment=True, cache=self.opt.cache_images,
                                                          rect=self.opt.rect, rank=self.rank,
                                                          world_size=self.opt.world_size, workers=self.opt.workers,
                                                          image_weights=self.opt.image_weights)

        # max label class
        self.max_label_class = np.concatenate(self.dataset.labels, 0)[:, 0].max()
        self.num_batches = len(self.dataloader)  # number of batches
        assert self.max_label_class < self.nc, f'Label class {self.max_label_class} exceeds nc={self.nc} in ' \
                                               f'{self.opt.data}. Possible class labels are 0-{self.nc - 1}'

        # Process 0
        if self.rank in [-1, 0]:
            self.ema.updates = self.start_epoch * self.num_batches // self.accumulate  # set EMA updates
            self.test_loader = create_dataloader(self.test_path, self.test_img_size,
                                                 self.total_batch_size, self.grid_size,
                                                 self.opt, hyp=self.hyp,
                                                 cache=self.opt.cache_images and not self.opt.notest, rect=True,
                                                 rank=-1, world_size=self.opt.world_size,
                                                 workers=self.opt.workers, pad=0.5)[0]

            if not self.opt.resume:
                labels = np.concatenate(self.dataset.labels, 0)
                c = torch.tensor(labels[:, 0])  # classes
                # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
                # model._initialize_biases(cf.to(device))
                # if plots:
                #     plot_labels(labels, save_dir, loggers)
                #     if tb_writer:
                #         tb_writer.add_histogram('classes', c, 0)

                # Anchors
                if not self.opt.noautoanchor:
                    check_anchors(self.dataset, model=self.model, thr=hyp['anchor_t'], imgsz=self.img_size)

            # Model parameters
            self.hyp['cls'] *= self.nc / 80.  # scale coco-tuned hyp['cls'] to current dataset
            self.model.nc = self.nc  # attach number of classes to model
            self.model.hyp = hyp  # attach hyperparameters to model
            self.model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
            self.model.class_weights = labels_to_class_weights(self.dataset.labels, self.nc).to(device) * self.nc  # attach class weights
            self.model.names = self.names

            # Start training
            self.start_time = time.time()
            self.nw = max(round(hyp['warmup_epochs'] * self.num_batches),
                     1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
            # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
            self.maps = np.zeros(self.nc)  # mAP per class
            self.results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
            self.scheduler.last_epoch = self.start_epoch - 1  # do not move
            self.scaler = amp.GradScaler(enabled=self.cuda)
            logger.info(f'Image sizes {self.img_size} train, {self.test_img_size} test\n'
                        f'Using {self.dataloader.num_workers} dataloader workers\nLogging'
                        f' results to {self.save_dir}\nStarting training for {self.epochs} epochs...')

    def train_loop(self, epochs):
        logger.info(f'Starting training for {epochs} epochs...')
        for epoch in range(self.start_epoch, self.epochs):
            self.model.train()

            # Update image weights (optional)
            if self.opt.image_weights:
                # Generate indices
                if self.rank in [-1, 0]:
                    cw = self.model.class_weights.cpu().numpy() * (1 - self.maps) ** 2  # class weights
                    iw = labels_to_image_weights(self.dataset.labels, nc=self.nc, class_weights=cw)  # image weights
                    self.dataset.indices = random.choices(range(self.dataset.n), weights=iw, k=self.dataset.n)  # rand weighted idx
                if self.rank != -1:
                    indices = (
                        torch.tensor(self.dataset.indices) if self.rank == 0 else torch.zeros(self.dataset.n)).int()
                    dist.broadcast(indices, 0)
                    if self.rank != 0:
                        self.dataset.indices = indices.cpu().numpy()

            self.mean_loss = torch.zeros(5, device=self.device)  # mean losses
            if self.rank != -1:
                self.dataloader.sampler.set_epoch(epoch)
            pbar = enumerate(self.dataloader)
            logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
            if self.rank in [-1, 0]:
                pbar = tqdm(pbar, total=self.num_batches)

            self.optimizer.zero_grad()
            for i, (imgs, targets, paths, _) in pbar:  # batch -----------------------------------
                ni = i + self.num_batches * epoch  # number integrated batches (since train start)
                imgs = imgs.to(self.device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

                # Warmup
                if ni <= self.nw:
                    xi = [0, self.nw]  # x interp
                    # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                    accumulate = max(1, np.interp(ni, xi, [1, self.nominal_batch_size / self.total_batch_size]).round())
                    for j, x in enumerate(self.optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(ni, xi, [self.hyp['warmup_bias_lr'] if j == 2 else 0.0,
                                                     x['initial_lr'] * self.lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [self.hyp['warmup_momentum'], self.hyp['momentum']])

                # Multi-scale
                gs = self.grid_size
                if self.opt.multi_scale:
                    sz = random.randrange(self.img_size * 0.5, self.img_size * 1.5 + gs) // gs * gs  # size
                    sf = sz / max(imgs.shape[2:])  # scale factor
                    if sf != 1:
                        ns = [math.ceil(x * sf / gs) * gs for x in
                              imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                        imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

                # Forward
                with amp.autocast(enabled=self.cuda):
                    pred = self.model(imgs)  # forward
                    loss, loss_items = compute_loss(pred, targets.to(self.device),
                                                    self.model)  # loss scaled by batch_size
                    if self.rank != -1:
                        loss *= self.opt.world_size  # gradient averaged between devices in DDP mode

                # Backward
                self.scaler.scale(loss).backward()

                # Optimize
                if ni % self.accumulate == 0:
                    self.scaler.step(self.optimizer)  # optimizer.step
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    if self.ema:
                        self.ema.update(self.model)

                # Print
                if self.rank in [-1, 0]:
                    mloss = (self.mean_loss * i + loss_items) / (i + 1)  # update mean losses
                    mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                    s = ('%10s' * 2 + '%10.4g' * 7) % (
                        '%g/%g' % (epoch, self.epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                    pbar.set_description(s)
                # end batch -----------------------------------------------------------------------------------------
                    # Scheduler
                lr = [x['lr'] for x in self.optimizer.param_groups]  # for tensorboard
                self.scheduler.step()

                # DDP process 0 or single-GPU
                if self.rank in [-1, 0] and epoch > 20:
                    # mAP
                    if self.ema:
                        self.ema.update_attr(self.model,
                                        include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
                    final_epoch = epoch + 1 == epochs
                    if not self.opt.notest or final_epoch:  # Calculate mAP
                        self.results, maps, times = test.test(self.opt.data,
                                                         batch_size=self.total_batch_size,
                                                         imgsz=self.test_img_size,
                                                         model=self.ema.ema,
                                                         single_cls=self.opt.single_cls,
                                                         dataloader=self.test_loader,
                                                         save_dir=self.save_dir,
                                                         plots=False,
                                                         log_imgs=self.opt.log_imgs if wandb else 0)

                    # Write
                    with open(self.results_file, 'a') as f:
                        f.write(
                            s + '%10.4g' * 7 % self.results + '\n')  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
                    if len(self.opt.name) and self.opt.bucket:
                        os.system(
                            f'gsutil cp {self.results_file} gs://{self.opt.bucket}/results/results{self.opt.name}.txt')
                        # sync results to cloud

                    # Log
                    tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                            'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                            'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                            'x/lr0', 'x/lr1', 'x/lr2']  # params
                    for x, tag in zip(list(self.mean_loss[:-1]) + list(self.results) + lr, tags):
                        if self.tb_writer:
                            self.tb_writer.add_scalar(tag, x, epoch)  # tensorboard
                        if wandb:
                            wandb.log({tag: x})  # W&B

                    # Update best mAP
                    fi = fitness(
                        np.array(self.results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
                    if fi > self.best_fitness:
                        best_fitness = fi

                    # Save model
                    save = (not self.opt.nameopt.nosave) or (final_epoch and not self.opt.nameopt.evolve)
                    if save:
                        with open(self.opt.nameresults_file, 'r') as f:  # create checkpoint
                            ckpt = {'epoch': epoch,
                                    'best_fitness': self.opt.namebest_fitness,
                                    'training_results': f.read(),
                                    'model': self.opt.nameema.ema,
                                    'optimizer': None if final_epoch else self.optimizer.state_dict(),
                                    'wandb_id': self.wandb_run.id if wandb else None}

                        # Save last, best and delete
                        torch.save(ckpt, self.last)
                        if self.best_fitness == fi:
                            torch.save(ckpt, self.best)
                        del ckpt
                # end epoch ----------------------------------------------------------------------------------------------------
            # end training

            if self.rank in [-1, 0]:
                # Strip optimizers
                final = self.best if self.best.exists() else self.last  # final model
                for f in [self.last, self.best]:
                    if f.exists():
                        strip_optimizer(f)  # strip optimizers
                if self.opt.bucket:
                    os.system(f'gsutil cp {final} gs://{self.opt.bucket}/weights')  # upload


                # Test best.pt
                logger.info(
                    f'{epoch - self.start_epoch + 1} epochs completed in'
                    f' {(time.time() - self.start_time) / 3600} hours.\n')

                if self.opt.data.endswith('coco.yaml') and self.nc == 80:  # if COCO
                    for conf, iou, save_json in ([0.25, 0.45, False], [0.001, 0.65, True]):  # speed, mAP tests
                        self.results, _, _ = test.test(self.opt.data,
                                                       batch_size=self.total_batch_size,
                                                       imgsz=self.test_img_size,
                                                       conf_thres=conf,
                                                       iou_thres=iou,
                                                       model=attempt_load(final, self.device).half(),
                                                       single_cls=self.opt.single_cls,
                                                       dataloader=self.test_loader,
                                                       save_dir=self.save_dir,
                                                       save_json=save_json,
                                                       plots=False)

            else:
                dist.destroy_process_group()

            self.wandb.run.finish() if self.wandb and self.wandb.run else None
            torch.cuda.empty_cache()
            return self.results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='models/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/widerface.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[800, 800], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', default=False, help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--log-imgs', type=int, default=16, help='number of images for W&B logging, max 100')
    parser.add_argument('--log-artifacts', action='store_true', help='log artifacts, i.e. final trained model')
    parser.add_argument('--workers', type=int, default=4, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()

    # Set DDP variables
    opt.total_batch_size = opt.batch_size
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)
    if opt.global_rank in [-1, 0]:
        check_git_status()

    # Resume
    if opt.resume:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))  # replace
        opt.cfg, opt.weights, opt.resume = '', ckpt, True
        logger.info('Resuming training from %s' % ckpt)
    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        opt.name = 'evolve' if opt.evolve else opt.name
        opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps
        if 'box' not in hyp:
            warn('Compatibility: %s missing "box" which was renamed from "giou" in %s' %
                 (opt.hyp, 'https://github.com/ultralytics/yolov5/pull/1120'))
            hyp['box'] = hyp.pop('giou')

    # Train
    logger.info(opt)
    if not opt.evolve:
        tb_writer = None  # init loggers
        if opt.global_rank in [-1, 0]:
            logger.info(f'Start Tensorboard with "tensorboard --logdir {opt.project}", view at http://localhost:6006/')
            tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
        trainer = Trainer(hyp, opt, device, tb_writer, wandb)
        results = trainer.train_loop(opt.epochs)  # train

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0)}  # image mixup (probability)

        assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
        opt.notest, opt.nosave = True, True  # only test/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        yaml_file = Path(opt.save_dir) / 'hyp_evolved.yaml'  # save best result here
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

        for _ in range(300):  # generations to evolve
            if Path('evolve.txt').exists():  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            trainer = Trainer(hyp.copy(), opt, device, wandb=wandb)
            results = trainer.train_loop(opt.epochs)  # train

            # Write mutation results
            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        # Plot results
        plot_evolution(yaml_file)
        print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
              f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')
