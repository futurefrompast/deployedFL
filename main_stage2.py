import argparse
import os
import random
import shutil
import math
import time
import copy
import warnings
from enum import Enum
from collections import OrderedDict
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset

from lib.models.swin import SwinTransformer
from lib.models.modules.parallel import DataParallel
from lib.utils import save_checkpoint, accuracy, ProgressMeter, AverageMeter, Summary
from lib.utils import cross_entropy
from lib.datasets.multiview_dataset import MultiViewGenerator
from lib.utils_meta import gradient_update_bn_parameters, gradient_update_other_parameters, \
                           update_client_parameters, communication


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='swin-t',
                    help='model architecture: (default: swin-t)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")

parser.add_argument('--train_dir', default=None, type=str,
                    help='training dataset.')
parser.add_argument('--val_dir', default=None, type=str,
                    help='validation dataset.')
parser.add_argument('--log_dir', default=None, type=str,
                    help='validation dataset.')

parser.add_argument('--lr_inner', default=0.001, type=float,
                    help='inner learning rate', dest='lr_inner')
parser.add_argument('--coeff_inner', default=0.1, type=float,
                    help='inner loss coefficient', dest='coeff_inner')
parser.add_argument('--num_accumulation_steps', default=1, type=int,
                    help='number of gradient accumulation steps')

parser.add_argument('--alpha_server', type=float, default=0.99, required=False, help='alpha_server')
parser.add_argument('--alpha_client', type=float, default=0.99, required=False, help='alpha_client')
parser.add_argument('--alpha_ma', type=float, default=0.99, required=False, help='alpha_ma')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to pretrained checkpoint (default: none)')

best_acc1 = 0


def main():
    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
        if ngpus_per_node == 1 and args.dist_backend == "nccl":
            warnings.warn("nccl backend >=2.5 requires GPU count>1, see https://github.com/NVIDIA/nccl/issues/103 perhaps use 'gloo'")
    else:
        ngpus_per_node = 1

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    #model = models.__dict__[args.arch]()
    model = SwinTransformer(
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7
                )
    client_model = copy.deepcopy(model) 

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    else:
        #model = torch.nn.DataParallel(model).cuda()
        model = DataParallel(model).cuda()
        client_model = DataParallel(client_model).cuda()

    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)

    # Data loading code
    if args.dummy:
        print("=> Dummy data is used!")
        train_dataset = datasets.FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor())
        val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())
    else:
        #traindir = os.path.join(args.data, 'train')
        #valdir = os.path.join(args.data, 'val')
        traindir = args.train_dir
        valdir = args.val_dir
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        transform_train = transforms.Compose([
                transforms.TrivialAugmentWide(interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        train_dataset = datasets.ImageFolder(
            traindir, 
            transform=MultiViewGenerator(transform_train, n_views=4)
        )

        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

    # Only subset (odd, even)
    train_dataset = torch.utils.data.Subset(train_dataset, list(range(0, len(train_dataset), 2)))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    steps_per_epoch = math.ceil(len(train_loader) / args.num_accumulation_steps)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, steps_per_epoch=steps_per_epoch, epochs=args.epochs)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch'] - 1
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            if args.gpu is None:
                checkpoint = torch.load(args.pretrained)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.pretrained, map_location=loc)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pretrained, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))


    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch+1, args.epochs+1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, client_model, criterion, optimizer, scheduler, epoch, device, args)

        # evaluate on validation set
        if epoch % 10 == 0 and epoch != 0: 
            acc1 = validate(val_loader, model, criterion, args)
            
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % ngpus_per_node == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'state_dict_ema': client_model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                    'scheduler' : scheduler.state_dict()
                }, is_best, log_dir=args.log_dir)

def train(train_loader, model_server, model_client, criterion, optimizer, scheduler, epoch, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses_server_outer = AverageMeter('Loss(ser-o)', ':.4e')
    losses_server_inner = AverageMeter('Loss(ser-i)', ':.4e')
    losses_client_outer = AverageMeter('Loss(cli-o)', ':.4e')
    losses_client_inner = AverageMeter('Loss(cli-i)', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses_server_outer, losses_server_inner, losses_client_outer, losses_client_inner, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model_server.train()
    model_client.train()

    end = time.time()
    for i, (images, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = [image.to(device, non_blocking=True) for image in images]
        labels = [labels.to(device, non_blocking=True) for _ in images]

        images_server_inner = images[0]
        labels_server_inner = labels[0]
        images_client_inner = images[1]
        labels_client_inner = labels[1]
        images_server_outer = images[2]
        labels_server_outer = labels[2]
        images_client_outer = images[3]
        labels_client_outer = labels[3]

        params_client = OrderedDict()
        params_client.update(OrderedDict(model_client.meta_named_parameters()))
        params_server = OrderedDict()
        params_server.update(OrderedDict(model_server.meta_named_parameters()))

        # Step 1. client update (norm)
        outputs_client_inner1 = model_client(images_client_inner, params=params_client)
        outputs_client_inner2 = model_client(images_server_inner, params=params_client)
        loss_client_inner = criterion(outputs_client_inner1, labels_client_inner)
        loss_sim_client_inner = cross_entropy(outputs_client_inner1, outputs_client_inner2)
        loss_sim_client_inner += cross_entropy(outputs_client_inner2, outputs_client_inner1)

        params_client = gradient_update_bn_parameters(
                                loss_client_inner,
                                params=params_client,
                                step_size=args.lr_inner, 
                                first_order=False)

        # Step 2. aggregate client params to server
        params_server = communication(params_server, 
                                      params_client, 
                                      alpha_server=args.alpha_server,
                                      alpha_client=args.alpha_client,
                                      side="server")
        
        # Step 3. server update (others)
        outputs_server_inner1 = model_server(images_server_inner, params=params_server)
        outputs_server_inner2 = model_server(images_client_inner, params=params_server)
        loss_server_inner = criterion(outputs_server_inner1, labels_server_inner)
        loss_sim_server_inner = cross_entropy(outputs_server_inner1, outputs_server_inner2)
        loss_sim_server_inner += cross_entropy(outputs_server_inner2, outputs_server_inner1)

        params_server = gradient_update_other_parameters(
                                loss_server_inner,
                                params=params_server, 
                                step_size=args.lr_inner, 
                                first_order=False)

        outputs_server_outer1 = model_server(images_server_outer, params=params_server)
        outputs_server_outer2 = model_server(images_client_outer, params=params_server)
        loss_server_outer = criterion(outputs_server_outer1, labels_server_outer) \
                            + criterion(outputs_server_outer2, labels_server_outer)
        loss_sim_server_outer = cross_entropy(outputs_server_outer1, outputs_server_outer2) \
                            + cross_entropy(outputs_server_outer2, outputs_server_outer1)

        # Step 4. client inference
        params_client = communication(params_server,
                                      params_client,
                                      alpha_server=args.alpha_server,
                                      alpha_client=args.alpha_client,
                                      side="client")

        outputs_client_outer1 = model_client(images_client_outer, params=params_client)
        outputs_client_outer2 = model_client(images_server_outer, params=params_client)
        loss_client_outer = criterion(outputs_client_outer1, labels_client_outer) \
                            + criterion(outputs_client_outer2, labels_client_outer)
        loss_sim_client_outer = cross_entropy(outputs_client_outer1, outputs_client_outer2) \
                            + cross_entropy(outputs_client_outer2, outputs_client_outer1)

        loss = loss_server_outer + loss_client_outer + loss_sim_server_outer + loss_sim_client_outer + \
                    loss_server_inner + loss_sim_server_inner + loss_client_inner + loss_sim_client_inner
        loss = loss / float(args.num_accumulation_steps)

        acc1, acc5 = accuracy(outputs_server_outer1, labels[0], topk=(1, 5))
        losses_server_outer.update(loss_server_outer.item(), images[0].size(0))
        losses_client_outer.update(loss_client_outer.item(), images[0].size(0))
        losses_server_inner.update(loss_server_inner.item(), images[0].size(0))
        losses_client_inner.update(loss_client_inner.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        loss.backward()
        update_client_parameters(model_server, model_client, args.alpha_ma)

        if ((i + 1) % args.num_accumulation_steps == 0) or (i + 1 == len(train_loader)):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)


def validate(val_loader, model, criterion, args):

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None and torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.backends.mps.is_available():
                    images = images.to('mps')
                    target = target.to('mps')
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    return top1.avg


if __name__ == '__main__':
    main()
