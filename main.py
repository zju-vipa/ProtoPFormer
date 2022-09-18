
   
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import random
import torch
import logging
import os
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path

import torchvision
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from tools.create_optimizer import create_optimizer
from tools.create_scheduler import create_scheduler

from timm.utils import NativeScaler, get_state_dict, ModelEma

import tools.utils as utils
import protopformer
from tools.engine_proto import train_one_epoch, evaluate
from tools.preprocess import mean, std
from tools.datasets import build_dataset
from tools.utils import str2bool


def get_args_parser():
    parser = argparse.ArgumentParser('Vision Transformer KD training and evaluation script',
                        add_help=False)
    parser.add_argument('--batch_size', default=256, type=int)
    # parser.add_argument('--distill', type=bool, default=False)
    parser.add_argument('--distillw', type=float, default=0.5, help='distill rate (default: 0.5)')
    parser.add_argument('--enable_smoothing', type=bool, default=False)
    parser.add_argument('--enable_mixup', type=bool, default=False)
    parser.add_argument('--w_dis_token', type=bool, default=False)

    # ProtoPFormer
    parser.add_argument('--base_architecture', type=str, default='deit_tiny_patch16_224')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--prototype_shape', nargs='+', type=int, default=[2000, 192, 1, 1])
    parser.add_argument('--prototype_activation_function', type=str, default='log')
    parser.add_argument('--add_on_layers_type', type=str, default='regular')
    parser.add_argument('--baseline_path', type=str, default=None)
    parser.add_argument('--reserve_layers', nargs='+', type=int, default=[])
    parser.add_argument('--reserve_token_nums', nargs='+', type=int, default=[])
    parser.add_argument('--use_global', type=str2bool, default=False)
    parser.add_argument('--use_ppc_loss', type=str2bool, default=False)
    parser.add_argument('--ppc_cov_thresh', type=float, default=1.)
    parser.add_argument('--ppc_mean_thresh', type=float, default=2.)
    parser.add_argument('--global_coe', type=float, default=0.5)
    parser.add_argument('--global_proto_per_class', type=int, default=5)
    parser.add_argument('--ppc_cov_coe', type=float, default=0.1)
    parser.add_argument('--ppc_mean_coe', type=float, default=0.5)

    parser.add_argument('--data_path', type=str, default='./datasets/cub200_cropped/')

    parser.add_argument('--features_lr', type=float, default=1e-4)
    parser.add_argument('--add_on_layers_lr', type=float, default=3e-3)
    parser.add_argument('--prototype_vectors_lr', type=float, default=3e-3)
    parser.add_argument('--joint_lr_step_size', type=int, default=5)

    parser.add_argument('--coefs_crs_ent', type=float, default=1)
    parser.add_argument('--coefs_clst', type=float, default=0.8)
    parser.add_argument('--coefs_sep', type=float, default=-0.08)
    parser.add_argument('--coefs_l1', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=40)

    # Model parameters
    parser.add_argument('--model', default='deit_tiny_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train') 
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--save_ep_freq', default=400, type=int, help='save epoch frequency')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model_ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.0, help='Label smoothing (default: 0.0)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    """
    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    """

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data_set', default='CIFAR100', 
    choices=['CUB2011U', 'Car', 'Dogs',],
    # choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='output_kd/test/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=1028, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser


def get_outlog(args):
    if args.eval: # evaluation only
        logfile_dir = os.path.join(args.output_dir, "eval-logs")
    else: # training
        logfile_dir = os.path.join(args.output_dir, "train-logs")
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    tb_dir = os.path.join(args.output_dir, "tf-logs")
    tb_log_dir = os.path.join(tb_dir, args.model+ "_" + args.data_set)
    os.makedirs(logfile_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)
    tb_writer = SummaryWriter(
        log_dir=os.path.join(
            tb_dir,
            args.model+ "_" + args.data_set
        ),
        flush_secs=1
    )
    logger = utils.get_logger(
        level=logging.INFO,
        mode="w",
        name=None,
        logger_fp=os.path.join(
            logfile_dir,
            args.model+ "_" + args.data_set + ".log"
        )
    )

    return tb_writer, logger


def get_model(args, model_name):
    if model_name in ["ResNet50", "ResNet101", "ResNet18", "Vgg16"]:
        if model_name == "ResNet50":
            model = torchvision.models.resnet50(pretrained=False, num_classes=args.nb_classes)
        elif model_name == "ResNet101":
            model = torchvision.models.resnet101(pretrained=False, num_classes=args.nb_classes)
        elif model_name == "ResNet18":
            model = torchvision.models.resnet18(pretrained=False, num_classes=args.nb_classes)
        elif model_name == "Vgg16":
            model = torchvision.models.vgg16(pretrained=False, num_classes=args.nb_classes)

    else:
        # if not args.w_dis_token:
        model = create_model(
            model_name,
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
        )
    return model


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(args):
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    set_seed(seed)

    if args.enable_smoothing:
        args.smoothing = 0.1
    utils.init_distributed_mode(args)

    tb_writer, logger = get_outlog(args)

    logger.info("Start running with args: \n{}".format(args))
    logger.info("Distributed: {}".format(args.distributed))

    device = torch.device(args.device)

    # cudnn.benchmark = True

    # get dataloaders
    normalize = transforms.Normalize(mean=mean,
                                    std=std)

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    # dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    # dataset_val, _ = build_dataset(is_train=False, args=args)
    logger.info("Dataset num_classes: {}".format(args.nb_classes))
    logger.info("train {} test: {}".format(len(dataset_train), len(dataset_val)))

    # if True:  # args.distributed:
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    if args.enable_smoothing:
        assert args.smoothing > 0.0
        logger.info("Label smoothing is enabled, smoothing rate: {}".format(args.smoothing))
    elif not args.enable_smoothing:
        assert args.smoothing == 0
        logger.info("Label smoothing is not enabled, smoothing rate: {}".format(args.smoothing))
    if args.enable_mixup:
        mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
        if mixup_active:
            mixup_fn = Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing, num_classes=args.nb_classes)
            logger.info("mixup_fn smoothing rate: {}".format(mixup_fn.label_smoothing))
    else:
        assert args.mixup == 0.0
        logger.info("Mixup is not enabled")

    # logger.info(f"Creating model: {args.model}")
    model = protopformer.construct_PPNet(base_architecture=args.base_architecture,
                                pretrained=True, img_size=args.img_size,
                                prototype_shape=args.prototype_shape,
                                num_classes=args.nb_classes,
                                reserve_layers=args.reserve_layers,
                                reserve_token_nums=args.reserve_token_nums,
                                use_global=args.use_global,
                                use_ppc_loss=args.use_ppc_loss,
                                ppc_cov_thresh=args.ppc_cov_thresh,
                                ppc_mean_thresh=args.ppc_mean_thresh,
                                global_coe=args.global_coe,
                                global_proto_per_class=args.global_proto_per_class,
                                prototype_activation_function=args.prototype_activation_function,
                                add_on_layers_type=args.add_on_layers_type)

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    joint_optimizer_lrs = {'features': args.features_lr,
                        'add_on_layers': args.add_on_layers_lr,
                        'prototype_vectors': args.prototype_vectors_lr,}

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params: {}'.format(n_parameters))

    # timm.optim
    optimizer = create_optimizer(args, model_without_ddp, joint_optimizer_lrs=joint_optimizer_lrs)
    # optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, args)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    logger.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            logger.info("distributed, data_loader_train set epoch")
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model=model, criterion=criterion, data_loader=data_loader_train,
            optimizer=optimizer, device=device, epoch=epoch, loss_scaler=loss_scaler,
            max_norm=args.clip_grad, model_ema=model_ema, mixup_fn=mixup_fn,
            args=args, tb_writer=tb_writer, iteration=__global_values__["it"],
            # set_training_mode=args.finetune == ''  # keep in eval mode during finetuning
        )
        logger.info("Averaged stats:")
        logger.info(train_stats)
        __global_values__["it"] += len(data_loader_train)
        tb_writer.add_scalar("epoch/train_loss", train_stats["loss"], epoch)
        
        lr_scheduler.step(epoch)
        if args.output_dir:
            if (epoch+1) % args.save_ep_freq == 0:
                checkpoint_paths = [output_dir / 'checkpoints/checkpoint-{}.pth'.format(epoch)]
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)

        test_stats = evaluate(data_loader=data_loader_val, model=model, device=device, args=args)
        logger.info(test_stats)

        tb_writer.add_scalar("epoch/val_acc1", test_stats['acc1'], epoch)
        tb_writer.add_scalar("epoch/val_loss", test_stats['loss'], epoch)
        tb_writer.add_scalar("epoch/val_acc5", test_stats['acc5'], epoch)
        if args.use_global:
            tb_writer.add_scalar("epoch/global_acc1", test_stats['global_acc1'], epoch)
            tb_writer.add_scalar("epoch/local_acc1", test_stats['local_acc1'], epoch)

        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        if max_accuracy < test_stats["acc1"]:   # save the best
            checkpoint_paths = [output_dir / 'checkpoints/epoch-best.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        logger.info(log_stats)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    __global_values__ = dict(it=0)
    main(args)