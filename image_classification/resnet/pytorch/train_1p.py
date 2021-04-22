import argparse
import os
import random
import shutil
import time
import warnings
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

from apex import amp

BATCH_SIZE = 512
EPOCHS_SIZE = 100
TRAIN_STEP = 8000
LOG_STEP = 50

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--platform', type=str, default='npu',
                    help='set which type of device you want to use. gpu/npu')
parser.add_argument('--device-id', type=str, default='0',
                    help='set device id')
parser.add_argument('--data',
                    metavar='DIR',
                    default="./data",
                    help='path to dataset')
parser.add_argument('-a', '--arch',
                    metavar='ARCH',
                    default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')

# apex
parser.add_argument('--amp', default=False, action='store_true',
                    help='use amp to train the model')
parser.add_argument('--loss-scale', default=-1, type=float,
                    help='loss scale using in amp, default -1 means dynamic')
parser.add_argument('--amp-level', default='O2', type=str,
                    help='amp optimization level, O2 means almost FP16, O0 means FP32')

parser.add_argument('-j', '--workers',
                    default=32,
                    type=int,
                    metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs',
                    default=EPOCHS_SIZE,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size',
                    default=BATCH_SIZE,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate',
                    default=0.1,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay',
                    default=1e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq',
                    default=10,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('-e', '--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed',
                    default=None,
                    type=int,
                    help='seed for initializing training. ')
parser.add_argument('--warmup',
                    default=0,
                    type=int,
                    metavar='E',
                    help='number of warmup epochs')
parser.add_argument('--label-smoothing',
                    default=0.0,
                    type=float,
                    metavar='S',
                    help='label smoothing')
parser.add_argument('--optimizer-batch-size',
                    default=-1,
                    type=int,
                    metavar='N',
                    help=
                    'size of a total batch size, for simulating bigger batches using gradient accumulation')

best_acc1 = 0

def main():
    args = parser.parse_args()
    print(args)
    if args.platform == "gpu":
        device = torch.device('cuda:' + args.device_id)
        device_func = torch.cuda
    elif args.platform == "npu":
        device = torch.device('npu:' + args.device_id)
        device_func = torch.npu
    else:
        device = torch.device('cpu')
    print("Running on device {}".format(device))
    global CALCULATE_DEVICE
    CALCULATE_DEVICE = device

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    # Only 1 card, simply call main_worker function
    main_worker(args)

def main_worker(args):
    global best_acc1

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](zero_init_residual=True)
    
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            torch.nn.init.kaiming_normal_(layer.weight, a=math.sqrt(5), )
    
    model = model.to(CALCULATE_DEVICE)

    writer = SummaryWriter(args.save_dir)

    lr_policy = lr_cosine_policy(args.lr, args.warmup, args.epochs)


    # define loss function (criterion) and optimizer
    loss = nn.CrossEntropyLoss
    if args.label_smoothing > 0.0:
        loss = lambda: LabelSmoothing(args.label_smoothing)
    criterion = loss().to(CALCULATE_DEVICE)
    optimizer = torch.optim.SGD([
        {'params': [param for name, param in model.named_parameters() if name[-4:] == 'bias'], 'weight_decay': 0.0},
        {'params': [param for name, param in model.named_parameters() if name[-4:] != 'bias'], 'weight_decay': args.weight_decay}],
                                args.lr,
                                momentum=args.momentum)
    if args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp_level, loss_scale=args.loss_scale, verbosity=0)

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch, args)
        lr_policy(optimizer, 0, epoch)
        # evaluate on validation set
        acc1, acc5 = validate(val_loader, model, criterion, args)
        writer.add_scalar('acc1/val', acc1, epoch)
        writer.add_scalar('acc5/val', acc5, epoch)
        # train for one epoch
        loss_avg, fps = train(train_loader, model, criterion, optimizer, epoch, args)
        writer.add_scalar('loss/train', loss_avg, epoch)
        writer.add_scalar('example_per_sec/train', fps, epoch)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        
        # file_name = "checkpoint_npu{}".format(args.npu)
        # modeltmp = model.cpu()
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'arch': args.arch,
        #     'state_dict': modeltmp.state_dict(),
        #     # 'state_dict': model,
        #     'best_acc1': best_acc1.to("cpu"),
        #     # 'optimizer' : optimizer.state_dict(),
        # }, is_best.to("cpu"), file_name)
        # modeltmp.to(CALCULATE_DEVICE)
    writer.close()

def train(train_loader, model, criterion, optimizer, epoch, args):
    if args.optimizer_batch_size < 0:
        batch_size_multiplier = 1
    else:
        tbs = 1 * args.batch_size
        if args.optimizer_batch_size % tbs != 0:
            print("Warning: simulated batch size {} is not divisible by actual batch size {}"
                    .format(args.optimizer_batch_size, tbs))
        batch_size_multiplier = int(args.optimizer_batch_size / tbs)
        print("BSM: {}".format(batch_size_multiplier))

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    optimizer.zero_grad()
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(CALCULATE_DEVICE, non_blocking=True)
        
        if args.label_smoothing == 0.0:
            if args.platform == "npu":
                target = target.to(torch.int32)
            
            target = target.to(CALCULATE_DEVICE, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        if args.label_smoothing > 0.0:
            if args.platform == "npu":
                target = target.to(torch.int32)
            
            target = target.to(CALCULATE_DEVICE, non_blocking=True)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        
        optimizer_step = ((i + 1) % batch_size_multiplier) == 0
        if optimizer_step:
            if batch_size_multiplier != 1:
                for param_group in optimizer.param_groups:
                    for param in param_group['params']:
                        param.grad /= batch_size_multiplier
            optimizer.step()
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

        if i == TRAIN_STEP:
            break
    fps = args.batch_size/batch_time.avg
    print("batch_size:", args.batch_size, 'Time: {:.3f}'.format(batch_time.avg), '* FPS@all {:.3f}'.format(
            fps))
    return losses.avg , fps

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(CALCULATE_DEVICE, non_blocking=True)
            if args.label_smoothing == 0.0:
                if args.platform == "npu":
                    target = target.to(torch.int32)

                target = target.to(CALCULATE_DEVICE, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            if args.label_smoothing > 0.0:
                if args.platform == "npu":
                    target = target.to(torch.int32)
                
                target = target.to(CALCULATE_DEVICE, non_blocking=True)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % LOG_STEP == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    return top1.avg, top5.avg

def save_checkpoint(state, is_best, filename='checkpoint'):
    filename2 = filename + ".pth.tar"
    torch.save(state, filename2)
    if is_best:
        shutil.copyfile(filename2, filename+'model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.start_count_index = 10

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.count == 0:
            self.batchsize = n
        
        self.val = val
        self.count += n
        if self.count > (self.start_count_index * self.batchsize):
            self.sum += val * n
            self.avg = self.sum / (self.count - self.start_count_index * self.batchsize)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.

        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1).to("cpu")
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean().to(CALCULATE_DEVICE)

def lr_policy(lr_fn, logger=None):
    if logger is not None:
        logger.register_metric('lr',
                               log.LR_METER(),
                               verbosity=dllogger.Verbosity.VERBOSE)

    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)

        if logger is not None:
            logger.log_metric('lr', lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr

def lr_cosine_policy(base_lr, warmup_length, epochs, logger=None):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn, logger=logger)

if __name__ == '__main__':
    main()