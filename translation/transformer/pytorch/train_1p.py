import torch
import time
from apex import amp
import os
import sys
import math
from utils import options, utils, criterions
from utils.ddp_trainer import DDPTrainer
from utils.meters import StopwatchMeter, TimeMeter
import data
from data import data_utils, load_dataset_splits
from models import build_model
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter

MAX = 2147483647
def _gen_seeds(shape):
    return np.random.uniform(1, MAX, size=shape).astype(np.float32)
seed_shape = (32 * 1024 * 12, )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

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

def main(args):
    print(args)

    if args.platform == "gpu":
        device = torch.device('cuda:' + args.device_id)
        device_func = torch.cuda
    elif args.platform == "npu":
        device = torch.device('npu:' + args.device_id)
        device_func = torch.npu

        # import any Python files in the optim/ directory
        # print(f"{os.path.dirname(__file__)}")
        
        # for file in os.listdir(os.path.dirname(__file__) + "/optim"):
        #     if file.endswith('.py') and not file.startswith('_'):
        #         module = file[:file.find('.py')]
        #         importlib.import_module('optim.' + module)
    else:
        device = torch.device('cpu')
    print("Running on device {}".format(device))
    args.device = device
    
    if args.max_tokens is None:
        args.max_tokens = 6000

    torch.manual_seed(args.seed)

    src_dict, tgt_dict = data_utils.load_dictionaries(args)
    datasets = load_dataset_splits(args, ['train', 'valid', 'test'], src_dict, tgt_dict)

    seed = _gen_seeds(seed_shape)
    seed = torch.from_numpy(seed)
    seed = seed.to(device)
    model = build_model(args, seed=seed)
    model = model.to(device)
    print('| num. model params: {}'.format(sum(p.numel() for p in model.parameters())))
    
    criterion = criterions.LabelSmoothedCrossEntropyCriterion(args).to(device)

    # optimizer = optim.build_optimizer(args, model.parameters())
    # Build trainer
    # trainer = DDPTrainer(args, model)
    print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))

    print('| training on {} devices'.format(args.distributed_world_size))
    print('| max sentences per NPU = {}'.format(args.max_sentences))

    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)

    if args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp_level, loss_scale=8, verbosity=0)

    writer = SummaryWriter(args.save_dir)

    epoch_itr = data.EpochBatchIterator(
        dataset=datasets[args.train_subset],
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences_valid,
        max_positions=args.max_positions,
        ignore_invalid_inputs=True,
        required_batch_size_multiple=8,
        seed=args.seed,
        num_shards=1,
        shard_id=0,
        max_positions_num=96,

    )

    # Train until the learning rate gets too small or model reaches target score
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_losses = [None]
    valid_subsets = args.valid_subset.split(',')
    run_summary = {'loss': float('inf'),
                   'val_loss': float('inf'),
                   'speed': 0,
                   'accuracy': 0}

    # max_update
    while epoch_itr.epoch < max_epoch:
        
        # train for one epoch
        train(args, datasets, epoch_itr, model, criterion, optimizer)

        if epoch_itr.epoch % args.validate_interval == 0:
            valid_losses = validate(args, datasets, epoch_itr, model, criterion, optimizer)

        writer.add_scalar('loss/val', valid_losses[0], epoch_itr.epoch)

    writer.close()
    train_meter.stop()
    print('| done training in {:.1f} seconds'.format(train_meter.sum))


def train(args, datasets, epoch_itr, model, criterion, optimizer):
    """Train the model for one epoch."""
    model.train()
    optimizer.zero_grad()
    itr = epoch_itr.next_epoch_itr()

    # update parameters every N batches
    if epoch_itr.epoch <= len(args.update_freq):
        update_freq = args.update_freq[epoch_itr.epoch - 1]
    else:
        update_freq = args.update_freq[-1]

    num_batches = len(epoch_itr)

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    sentence_s = AverageMeter('Sentence/s', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')

    progress = ProgressMeter(int(num_batches),
                             [batch_time, data_time, sentence_s, losses],
                             prefix = "Epoch: [{}]".format(epoch_itr.epoch))

    print("Update Frequence is :", str(update_freq))

    first_valid = args.valid_subset.split(',')[0]
    max_update = args.max_update or math.inf

    end = time.time()
    for i, sample in enumerate(itr):
        data_time.update(time.time() - end)
        # move sample to device
        sample = utils.move_to_device(args, sample)

        # calculate loss and sample size
        # src_tokens, src_lengths, prev_output_tokens
        # npu only accept int32 tensors
        if args.platform == "npu":
            sample['net_input']['src_tokens'] = sample['net_input']['src_tokens'].to(torch.int32)
            sample['net_input']['prev_output_tokens'] = sample['net_input']['prev_output_tokens'].to(torch.int32)
            sample['target'] = sample['target'].to(torch.int32)
        elif args.platform == "gpu":
            sample['net_input']['src_tokens'] = sample['net_input']['src_tokens'].to(torch.int64)
            sample['net_input']['prev_output_tokens'] = sample['net_input']['prev_output_tokens'].to(torch.int64)
            sample['target'] = sample['target'].to(torch.int64)

        logits, _ = model(sample['net_input']['src_tokens'], sample['net_input']['src_lengths'], sample['net_input']['prev_output_tokens'])
        target = sample['target']
        
        probs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        loss = criterion(probs, target)

        losses.update(loss.item() / sample['ntokens'] / math.log(2))

        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

        if i % 10 == 0:
            progress.display(i)

        batch_time.update(time.time() - end)
        end = time.time()

    print("End of epoch, batch_size:", args.max_sentences, 'Time: {:.3f}'.format(batch_time.avg), ' Sentence/s@all {:.3f}'.format(
        args.max_sentences / batch_time.avg))


def validate(args, datasets, model, criterion, optimizer):
    """Evaluate the model on the validation set(s) and return the losses."""
    model.eval()

    # Initialize data iterator
    itr = data.EpochBatchIterator(
        dataset=datasets[subset],
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences_valid,
        max_positions=args.max_positions,
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=8,
        seed=args.seed,
        num_shards=1,
        shard_id=0,
        max_positions_num=1024,
    ).next_epoch_itr(shuffle=False)

    num_batches = len(itr)
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        int(num_batches),
        [batch_time, losses],
        prefix='Test: ')

    for i, sample in enumerate(itr):
        # move sample to device
        sample = utils.move_to_device(args, sample)
        with torch.no_grad():
            if args.platform == "npu":
                sample['net_input']['src_tokens'] = sample['net_input']['src_tokens'].to(torch.int32)
                sample['net_input']['prev_output_tokens'] = sample['net_input']['prev_output_tokens'].to(torch.int32)
                sample['target'] = sample['target'].to(torch.int32)
            elif args.platform == "gpu":
                sample['net_input']['src_tokens'] = sample['net_input']['src_tokens'].to(torch.int64)
                sample['net_input']['prev_output_tokens'] = sample['net_input']['prev_output_tokens'].to(torch.int64)
                sample['target'] = sample['target'].to(torch.int64)
            logits, _ = model(sample['net_input']['src_tokens'], sample['net_input']['src_lengths'], sample['net_input']['prev_output_tokens'])
            target = sample['target']
    
            probs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            loss = criterion(probs, target)
            losses.update(loss.item() / sample['ntokens'] / math.log(2))

            if i % 10 == 0:
                progress.display(i)

    print(f'Validation loss: {losses.avg}')
    return losses.avg


if __name__ == '__main__':
    parser = options.get_training_parser()
    ARGS = options.parse_args_and_arch(parser)

    main(ARGS)