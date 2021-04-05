import collections
import itertools
import os
import math
import torch
import torch.npu
import time
import ctypes

import sys
import threading

from copy import deepcopy
from utils import distributed_utils, options, utils
from utils.ddp_trainer import DDPTrainer
from utils.meters import StopwatchMeter, TimeMeter
import data
from data import tokenizer, dictionary, data_utils, load_dataset_splits
from models import build_model
import numpy as np
import dllogger as DLLogger
from utils.log_helper import AggregatorBackend, setup_logger

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
    setup_logger(args)

    if args.platform == "gpu":
        device = torch.device('cuda:' + str(args.device_id))
        device_func = torch.cuda
        torch.cuda.set_device(device)
    elif args.platform == "npu":
        device = torch.device('npu:' + str(args.device_id))
        device_func = torch.npu
        torch.npu.set_device(device)
    else:
        device = torch.device('cpu')
    args.device = device
    print("Running on device {}".format(device))

    torch.manual_seed(args.seed)

    src_dict, tgt_dict = data_utils.load_dictionaries(args)
    datasets = data_utils.get_dummy_batch(args.max_sentences, src_dict, tgt_dict)

    seed = _gen_seeds(seed_shape)
    seed = torch.from_numpy(seed)
    seed = seed.to(device)
    model = build_model(args, seed=seed)
    params = sum(p.numel() for p in model.parameters())

    # Build trainer
    trainer = DDPTrainer(args, model)
    print('model {}, criterion {}'.format(args.arch, trainer.criterion.__class__.__name__))

    print('training on {} NPUs'.format(args.distributed_world_size))
    print('max sentences per NPU = {}'.format(args.max_sentences))

    # warm up
    for i in range(10):
        train(args, trainer, datasets)
    
    start_event = device_func.Event(enable_timing=True)
    end_event = device_func.Event(enable_timing=True)
    start_event.record()

    for i in range(100):
        train(args, trainer, datasets)

    end_event.record()
    device_func.synchronize()
    elapsed_time = start_event.elapsed_time(end_event) / 1000
    example_per_sec = args.max_sentences * args.num_iterations / elapsed_time
    print(f"time: {elapsed_time:.3f}")
    print(f"params: {params}")
    print(f"example per sec: {example_per_sec}")
    

def train(args, trainer, datasets):
    """Train the model."""
    for i, sample in enumerate(datasets):
        loss = trainer.train_step(sample, update_params=False)


def validate(args, trainer, datasets, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""
    # Reset value iterations counter
    trainer._num_val_iterations = 0

    valid_losses = []
    for subset in subsets:

        if len(subsets) > 1:
            print('Validating on \'{}\' subset'.format(subset))

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

        # reset validation loss meters
        DLLogger.flush()

        subset_losses = []
        for sample in itr:
            loss = trainer.valid_step(sample)
            subset_losses.append(loss)
        subset_loss = sum(subset_losses) / len(subset_losses)

        DLLogger.flush()

        valid_losses.append(subset_loss)
        print(f'Validation loss on subset {subset}: {subset_loss}')

    return valid_losses


def save_checkpoint(args, trainer, epoch_itr, val_loss):
    if args.no_save or not distributed_utils.is_master(args):
        return
    epoch = epoch_itr.epoch
    end_of_epoch = epoch_itr.end_of_epoch()
    updates = trainer.get_num_updates()

    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds['checkpoint{}.pt'.format(epoch)] = (
            end_of_epoch and not args.no_epoch_checkpoints and
            epoch % args.save_interval == 0
    )
    checkpoint_conds['checkpoint_{}_{}.pt'.format(epoch, updates)] = (
            not end_of_epoch and args.save_interval_updates > 0 and
            updates % args.save_interval_updates == 0
    )
    checkpoint_conds['checkpoint_best.pt'] = (
            val_loss is not None and
            (not hasattr(save_checkpoint, 'best') or val_loss < save_checkpoint.best)
    )
    checkpoint_conds['checkpoint_last.pt'] = True  # keep this last so that it's a symlink

    prev_best = getattr(save_checkpoint, 'best', val_loss)
    if val_loss is not None:
        save_checkpoint.best = min(val_loss, prev_best)
    extra_state = {
        'best': save_checkpoint.best,
        'train_iterator': epoch_itr.state_dict(),
        'val_loss': val_loss,
    }
    extra_state.update(save_checkpoint.extra_items)

    checkpoints = [os.path.join(args.save_dir, 'checkpoints', fn) for fn, cond in checkpoint_conds.items() if cond]
    if len(checkpoints) > 0:
        for cp in checkpoints:
            trainer.save_checkpoint(cp, extra_state)

    if not end_of_epoch and args.keep_interval_updates > 0:
        # remove old checkpoints; checkpoints are sorted in descending order
        checkpoints = utils.checkpoint_paths(os.path.join(args.save_dir, 'checkpoints'),
                                             pattern=r'checkpoint_\d+_(\d+)\.pt')
        for old_chk in checkpoints[args.keep_interval_updates:]:
            os.remove(old_chk)


def add_extra_items_to_checkpoint(dict):
    if not hasattr(save_checkpoint, 'extra_items'):
        save_checkpoint.extra_items = {}
    save_checkpoint.extra_items.update(dict)


def load_checkpoint(args, trainer, epoch_itr):
    """Load a checkpoint and replay dataloader to match."""
    os.makedirs(os.path.join(args.save_dir, 'checkpoints'), exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, 'checkpoints', args.restore_file)
    if os.path.isfile(checkpoint_path):
        extra_state = trainer.load_checkpoint(checkpoint_path)
        if extra_state is not None:
            # replay train iterator to match checkpoint
            epoch_itr.load_state_dict(extra_state['train_iterator'])

            print('| loaded checkpoint {} (epoch {} @ {} updates)'.format(
                checkpoint_path, epoch_itr.epoch, trainer.get_num_updates()))

            trainer.lr_step(epoch_itr.epoch)
            trainer.lr_step_update(trainer.get_num_updates())
            if 'best' in extra_state:
                save_checkpoint.best = extra_state['best']


if __name__ == '__main__':
    parser = options.get_training_parser()
    ARGS = options.parse_args_and_arch(parser)

    main(ARGS)