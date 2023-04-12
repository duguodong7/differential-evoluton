# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
T2T-ViT training and evaluating script
This script is modified from pytorch-image-models by Ross Wightman (https://github.com/rwightman/pytorch-image-models/)
It was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)
"""
import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from spikingjelly.clock_driven import functional

# import models

import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP
import sade
from sade import model_dict_to_vector, model_vector_to_dict
import numpy as np
# from plot_loss import plot

from data import ImageDataset, IterableImageDataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
# from timm.data import ImageDataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.layers import convert_splitbn_model
from timm.models import load_checkpoint, create_model, resume_checkpoint
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler

import vit_snn

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
_logger = logging.getLogger('train')
# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='T2T-ViT Training and Evaluating')

# Dataset / Model parameters
parser.add_argument('--data', default='/home/guodong/ImageNet',
                    help='path to dataset')
parser.add_argument('--data_wds', default='/home/guodong/imagenet_wds/shards',
                    help='path to dataset')
parser.add_argument('--model', default='resnet18', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--only_test', action='store_true', default=False,
                    help='if test only')
parser.add_argument('--eval_checkpoint', default='', type=str, metavar='PATH',
                    help='path to eval checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=1000, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--img_size', type=int, default=224, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=1, metavar='N',
                    help='ratio of validation batch size to training batch size (default: 1)')

# DE parameters
parser.add_argument('--de_batch_size', type=int, default=512, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--mini_batch_size', type=int, default=256, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--de_epochs', type=int, default=300, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--de_iters', type=int, default=2, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--popsize', type=int, default=10, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--pop_init_dir', default='/home/guodong/ImageNet',
                    help='path to dataset')
parser.add_argument('--log_dir', default='/root/declc_guodong/log_out/resnet50.txt',
                    help='path to dataset')

# Optimizer parameters
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='weight decay (default: 0.0001)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')

# Learning rate schedule parameters
parser.add_argument('--sched', default='multistep', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit')
parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--start_epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay_epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay_rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
parser.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
parser.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original or rand-m9-mstd0.5-inc1". (default: None)'),
parser.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--jsd', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                    help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='const',
                    help='Random erase mode (default: "const")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0.,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix', type=float, default=0.,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.,
                    help='Label smoothing (default: 0.1)')
parser.add_argument('--train_interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
parser.add_argument('--drop', type=float, default=0., metavar='PCT',
                    help='Dropout rate (default: 0.1)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=0., metavar='PCT',
                    help='Drop path rate (default: 0.1)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')

# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync_bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist_bn', type=str, default="reduce",
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--split_bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.99996,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log_interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery_interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('-j', '--workers', type=int, default=16, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('-j_val', '--val_workers', type=int, default=16, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--pin_mem', default=True,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--eval_metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
# logging.basicConfig(level=logging.DEBUG, filename='/root/declc_guodong/log_out/loss5.txt', filemode='a')

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    args, args_text = _parse_args()
    setup_default_logging(log_path=args.log_dir)
    logging.basicConfig(level=logging.DEBUG, filename=args.log_dir, filemode='a')
    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        print('distributed:', args.distributed)
        if args.distributed and args.num_gpu > 1:
            _logger.warning(
                'Using more than one GPU per process in distributed mode is not allowed.Setting num_gpu to 1.')
            args.num_gpu = 1

    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.num_gpu = 1
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
    assert args.rank >= 0

    if args.distributed:
        _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        _logger.info('Training with a single process on %d GPUs.' % args.num_gpu)

    torch.manual_seed(args.seed + args.rank)

    model = create_model(
        args.model,
        pretrained=False,
        drop_rate=0.,
        drop_path_rate=0.2,
        drop_block_rate=None)
    print("Creating model")
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")

    if args.local_rank == 0:
        _logger.info('Model %s created, param count: %d' %
                     (args.model, sum([m.numel() for m in model.parameters()])))

    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)

    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    use_amp = None
    if args.amp:
        # for backwards compat, `--amp` arg tries apex before native amp
        if has_apex:
            args.apex_amp = True
        elif has_native_amp:
            args.native_amp = True
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        _logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")

    model.cuda()
    if args.channels_last:
       model = model.to(memory_format=torch.channels_last)

    optimizer = create_optimizer(args, model)

    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if args.local_rank == 0:
            _logger.info('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    population = []
    for file in os.listdir(args.pop_init_dir):
       if len(population) >= args.popsize: break
       else: 
          if file.split('-')[0] == 'checkpoint':
               resume_path = os.path.join(args.pop_init_dir, file)
               resume_epoch = resume_checkpoint(model, resume_path, log_info=args.local_rank==0)-1
               solution = model_dict_to_vector(model).detach()
               population.append(solution)
               # eval_metrics = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)
    resume_epoch = None

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume=args.resume)

    if args.distributed:
        if args.sync_bn:
            assert not args.split_bn
            try:
                if has_apex and use_amp != 'native':
                    # Apex SyncBN preferred unless native amp is activated
                    model = convert_syncbn_model(model)
                else:
                    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                if args.local_rank == 0:
                    _logger.info(
                        'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                        'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')
            except Exception as e:
                _logger.error('Failed to enable Synchronized BatchNorm. Install Apex or Torch >= 1.1')
        if has_apex and use_amp != 'native':
            # Apex DDP preferred unless native amp is activated
            if args.local_rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[args.local_rank])  # can use device str in Torch >= 1.1
        # NOTE: EMA model does not need to be wrapped by DDP

    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        _logger.info('Scheduled epochs: {}'.format(num_epochs))

    train_dir = os.path.join(args.data, 'train')
    if not os.path.exists(train_dir):
        _logger.error('Training folder does not exist at: {}'.format(train_dir))
        exit(1)
    # dataset_train = ImageDataset(root=train_dir)
    dataset_de = ImageDataset(root=train_dir)
    # dataset_de = IterableImageDataset(root=args.data_wds, reader='wds/imagenet',
    #                                     split='imagenet-train-{000000..001281}.tar|1281167', 
    #                                     is_training=True, batch_size=args.mini_batch_size,
    #                                     seed=42, repeats=1)

    # dataset_eval = IterableImageDataset(root=args.data, reader='wds/imagenet',
    #                                     split='imagenet-val-{000000..000049}.tar|50000', 
    #                                     is_training=False, batch_size=args.batch_size,
    #                                     seed=42, repeats=1)


    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    if num_aug_splits > 1:
        # dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)
        dataset_de = AugMixDataset(dataset_de, num_splits=num_aug_splits)

    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']

    loader_de_args = dict(
        input_size=data_config['input_size'],
        batch_size=args.mini_batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        persistent_workers=False,
        worker_seeding='all',
     )

    eval_dir = os.path.join(args.data, 'val')
    if not os.path.isdir(eval_dir):
        eval_dir = os.path.join(args.data, 'validation')
        if not os.path.isdir(eval_dir):
            _logger.error('Validation folder does not exist at: {}'.format(eval_dir))
            exit(1)
    dataset_eval = ImageDataset(eval_dir)

    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size_multiplier * args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.val_workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )

    if args.jsd:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing).cuda()
    elif mixup_active:
        # smoothing is handled with mixup target transform
        train_loss_fn = SoftTargetCrossEntropy().cuda()
    elif args.smoothing:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing).cuda()
    else:
        train_loss_fn = nn.CrossEntropyLoss().cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()
    loss_fn = nn.CrossEntropyLoss().cuda()

    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    # score_lst = np.zeros(popsize)
    # population = []
    # get_pop = Get_Pop(model=model, popsize=num_epochs)
    
    if args.only_test:
        for i in range(10):
            val_metrics = validate(model, loader_eval, validate_loss_fn, args)
            _logger.info(f"Top-1,5 accuracy of the model is: {val_metrics['top1']:.3f}%, {val_metrics['top5']:.3f}%")
        return
        
    if args.eval_checkpoint:  # evaluate the model
        load_checkpoint(model, args.eval_checkpoint, args.model_ema)
        val_metrics = validate(model, loader_eval, validate_loss_fn, args)
        _logger.info(f"Top-1 accuracy of the model is: {val_metrics['top1']:.3f}%")
        return

    saver = None
    output_dir = ''
    if args.local_rank == 0:
        output_base = args.output if args.output else './output'
        exp_name = '-'.join([
            datetime.now().strftime("%Y%m%d-%H%M%S"),
            args.model,
            str(data_config['input_size'][-1])
        ])
        output_dir = get_outdir(output_base, 'train', exp_name)
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(
            model=model, optimizer=optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler,
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    eval_metrics_acc1 = torch.zeros(args.popsize).tolist()
    eval_metrics_acc5 = torch.zeros(args.popsize).tolist()
    eval_metrics_loss = torch.zeros(args.popsize).tolist()
    for i in range(args.popsize): #!!!
         solution = population[i]
         model_weights_dict = model_vector_to_dict(model=model, weights_vector=solution)
         model.load_state_dict(model_weights_dict)
         eval_metrics_temp = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)
         # eval_metrics_loss.append(eval_metrics_temp['loss'])
         # eval_metrics_temp = OrderedDict([('top1', 233), ('top5', 233)])
         eval_metrics_acc1[i] = round(eval_metrics_temp['top1'], 4)
         eval_metrics_acc5[i] = round(eval_metrics_temp['top5'], 4)
         eval_metrics_loss[i] = round(eval_metrics_temp['loss'], 4) 

    eval_metrics = OrderedDict([('top1', eval_metrics_acc1), ('top5', eval_metrics_acc5),('eval_loss',eval_metrics_loss)])

    for epoch in range(args.de_epochs):
       loader_de_args['worker_seeding'] = epoch + 233
       loader_de = create_loader(dataset_de, **loader_de_args)
       if args.distributed:
           loader_de.sampler.set_epoch(epoch)
       eval_metrics = DE_epoch(
           epoch, model, loader_de, population, args, saver, output_dir, loader_eval, validate_loss_fn, eval_metrics, amp_autocast=amp_autocast)
       if saver is not None:
           # save proper checkpoint with eval metric
           save_metric = eval_metrics[eval_metric]
           best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)
       # if (epoch%5 == 0 and epoch !=0):
       #     plot('./summary.csv')


    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


def DE_epoch(epoch, model, loader, population, args, saver, output_dir, loader_eval, validate_loss_fn, eval_metrics, amp_autocast=suppress):
     de_paras = [0.1, 0.1, 0.1, 0.7, 0.1, 0.1, 0.1, [], []]
     dim = len(model_dict_to_vector(model))
     bounds = 0.1
     loss_fn = nn.CrossEntropyLoss().cuda()
     # data_time_m = AverageMeter()
     # losses_m = AverageMeter()
     # top1_m = AverageMeter()
     # top5_m = AverageMeter()
     model.eval()

     def score_func(population):
          popsize = len(population)
          batch_time_m = AverageMeter()
          data_time_m = AverageMeter()
          acc1_all = torch.zeros(popsize).tolist()
          acc5_all = torch.zeros(popsize).tolist()
          loss_all = torch.zeros(popsize).tolist()#!!!
          end = time.time()
          torch.set_grad_enabled(False)
          for batch_idx, (input, target) in enumerate(loader):
               if batch_idx >= (args.de_batch_size//args.mini_batch_size): break
               data_time_m.update(time.time() - end)
               for i in range(0, popsize):
                    solution = population[i]
                    model_weights_dict = model_vector_to_dict(model=model, weights_vector=solution)
                    model.load_state_dict(model_weights_dict)
                    # model.eval()
                    # last_idx = len(loader) - 1
                    if not args.prefetcher:
                         input = input.cuda()
                         target = target.cuda()
                    if args.channels_last:
                         input = input.contiguous(memory_format=torch.channels_last)

                    # for item in range(input.size(0)//args.mini_batch_size):
                    #      mini_input = input[args.mini_batch_size*item: args.mini_batch_size*(item+1)]
                    #      mini_target = target[args.mini_batch_size*item: args.mini_batch_size*(item+1)]
                    if batch_idx==0 and args.local_rank < 1 and i == 0:
                         _logger.info('pop: {} input: {}'.format(i, input.flatten()[6000:6005]))
                    with amp_autocast():
                         output = model(input)
                    if isinstance(output, (tuple, list)):
                         output = output[0]
                    loss = loss_fn(output, target)
                    acc1, acc5 = accuracy(output, target, topk=(1, 5))
                    if args.distributed:
                         reduced_loss = reduce_tensor(loss.data, args.world_size)
                         acc1 = reduce_tensor(acc1, args.world_size)
                         acc5 = reduce_tensor(acc5, args.world_size)
                    # _logger.info('acc1: {}  acc5: {}'.format(acc1, acc5))
                    acc1_all[i] += acc1
                    acc5_all[i] += acc5
                    loss_all[i] += reduced_loss
               batch_time_m.update(time.time() - end)
               end = time.time()

          if args.local_rank == 0:
               print('data_time: {time1.val:.3f} ({time1.avg:.3f})  '
                     'batch_time: {time2.val:.3f} ({time2.avg:.3f})  '.format(
                                                       time1=data_time_m, time2=batch_time_m))                 
          score_lst_acc = [-i/(args.de_batch_size//args.mini_batch_size) for i in acc1_all]#!!!
          score_lst_loss = [i/(args.de_batch_size//args.mini_batch_size) for i in loss_all]#!!!
          return score_lst_acc,score_lst_loss#!!!

     score_lst_acc, score_lst = score_func(population)#score_lst contain the loss
     # best_score = -min(score_lst)
     bestidx = score_lst.index(min(score_lst))
     de_iter_acc = [round(-i.item(), 4) for i in score_lst_acc]
     de_iter_loss = [round(j.item(), 4) for j in score_lst]
     if args.local_rank == 0:
          _logger.info('de_iter:{}, best_score:{:>7.4f}, best_idx:{}, de_iter_loss: {}'.format(0, min(score_lst), bestidx, de_iter_loss))
     
     de_iter_dict = OrderedDict([('iter', 0), ('bestidx', bestidx), ('train_loss', de_iter_loss)])
     update_summary(
           epoch, de_iter_dict, eval_metrics, os.path.join(output_dir, 'summary.csv'),
           write_header=True)

     popsize = len(population)
     de_iter_time_m = AverageMeter()
     end = time.time()

     for de_iter in range(1, args.de_iters+1):
          population, score_lst, bestidx, de_paras, change_label = sade.evolve4(
                           score_func, epoch, bounds, dim, popsize, population, score_lst, de_paras)
          solution = population[bestidx]
          best_score = score_lst[bestidx]
          model_weights_dict = model_vector_to_dict(model=model, weights_vector=solution)
          model.load_state_dict(model_weights_dict)    

          de_iter_acc = [round(-i.item(), 4) for i in score_lst_acc]
          de_iter_loss = [round(j.item(), 4) for j in score_lst]
          if args.local_rank == 0:
               _logger.info('de_iter:{}, best_score:{:>7.4f}, best_idx:{}, de_iter_loss: {}'.format(de_iter, best_score, bestidx, de_iter_loss))
          # eval_metrics_loss=[]
          for i in range(popsize): #!!!
              if change_label[i] == 1:
                   solution = population[i]
                   model_weights_dict = model_vector_to_dict(model=model, weights_vector=solution)
                   model.load_state_dict(model_weights_dict)
                   eval_metrics_temp = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)
                   # eval_metrics_loss.append(eval_metrics_temp['loss'])
                   eval_metrics['top1'][i] = round(eval_metrics_temp['top1'], 4)
                   eval_metrics['top5'][i] = round(eval_metrics_temp['top5'], 4)
                   eval_metrics['eval_loss'][i] = round(eval_metrics_temp['loss'], 4)
              
          if args.local_rank == 0:
               _logger.info('eval_metrics_acc1: {}'.format(eval_metrics['top1']))

          # eval_metrics = OrderedDict([('loss', eval_metrics_loss), ('top1', eval_metrics_acc1), ('top5', eval_metrics_acc5)])
          # eval_metrics = OrderedDict([('top1', 70.), ('top5', 100)])
          torch.cuda.synchronize()
          de_iter_time_m.update(time.time() - end)
          end = time.time()
          
          if args.local_rank == 0:
               _logger.info(
                         'DE: {} [de_iter: {}]  '
                         'Acc@1: {top1:>7.4f}  '
                         'Acc@5: {top5:>7.4f}  '
                         'Iter_time: {de_iter_time.val:.3f}s, {rate:>7.2f}/s  '.format(
                             epoch, de_iter,
                             top1 = eval_metrics['top1'][bestidx],
                             top5 = eval_metrics['top5'][bestidx],
                             de_iter_time=de_iter_time_m,
                             rate= args.de_batch_size * args.world_size / de_iter_time_m.val))
          
          # torch.save(population,'/root/declc/pth/population.pth')
          de_iter_dict = OrderedDict([('iter', de_iter), ('bestidx', bestidx), ('train_loss', de_iter_loss)])
          update_summary(
                epoch, de_iter_dict, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                write_header=True)

     return eval_metrics


def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=''):
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            # if batch_idx > 1: break
            data_time_m.update(time.time() - end)
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            # if batch_idx==0 and args.local_rank == 0:#!!!
            #    _logger.info('validate, input: {}'.format(input.flatten()[6000:6005]))#!!!

            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]
            # reduced_loss_all = torch.zeros(popsize).tolist()
            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'DataTime: {data_time.val:.3f} ({data_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m, data_time=data_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics


if __name__ == '__main__':
    main()
