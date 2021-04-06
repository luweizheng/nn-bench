import collections
import itertools
import os
import math
import torch
import torch.optim as optim
import time
import ctypes
from apex import amp

import sys
import threading

from copy import deepcopy
from utils import options, utils, criterions
import data
from data import tokenizer, dictionary, data_utils, load_dataset_splits
from models import build_model
import numpy as np

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
    assert len(datasets) > 0, "随机生成数据为空！"
    sample = datasets[0]
    src_tokens = sample['net_input']['src_tokens'].to(args.device)
    src_lengths = sample['net_input']['src_lengths'].to(args.device)
    prev_output_tokens = sample['net_input']['prev_output_tokens'].to(args.device)
    target = sample['target'].to(args.device)

    if args.platform == "npu":
        src_tokens = src_tokens.to(torch.int32)
        src_lengths = src_lengths.to(torch.int32)
        prev_output_tokens = prev_output_tokens.to(torch.int32)
        target = target.to(torch.int32)

    seed = _gen_seeds(seed_shape)
    seed = torch.from_numpy(seed)
    seed = seed.to(device)
    model = build_model(args, seed=seed)
    model = model.to(args.device)
    params = sum(p.numel() for p in model.parameters())

    optimizer = optim.Adam(model.parameters())

    criterion = criterions.LabelSmoothedCrossEntropyCriterion(args).to(device)

    if args.amp:
        model, optimizer = amp.initialize(model, optimizer, 
                opt_level=args.amp_level if args.amp_level else 'O2', 
                loss_scale=8,
                verbosity=0)

    # Build trainer
    # trainer = DDPTrainer(args, model)
    print('model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))

    print('training on {} NPUs'.format(args.distributed_world_size))
    print('max sentences per NPU = {}'.format(args.max_sentences))

    # warm up
    for i in range(10):
        train(args, model, src_tokens, src_lengths, prev_output_tokens, target, criterion, optimizer)
    
    start_event = device_func.Event(enable_timing=True)
    end_event = device_func.Event(enable_timing=True)
    start_event.record()

    for i in range(100):
        train(args, model, src_tokens, src_lengths, prev_output_tokens, target, criterion, optimizer)

    end_event.record()
    device_func.synchronize()
    elapsed_time = start_event.elapsed_time(end_event) / 1000
    example_per_sec = args.max_sentences * args.num_iterations / elapsed_time
    print(f"time: {elapsed_time:.3f}")
    print(f"params: {params}")
    print(f"example per sec: {example_per_sec}")
    

def train(args, model, src_tokens, src_lengths, prev_output_tokens, target, criterion, optimizer):
    """Train the model."""
    model.train()

    logits, _ = model(src_tokens, src_lengths, prev_output_tokens)
    probs = torch.nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32)
    loss = criterion(probs, target)
    
    #loss = trainer.train_step(sample, update_params=False)

    if args.amp:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
    optimizer.step()

if __name__ == '__main__':
    parser = options.get_training_parser()
    ARGS = options.parse_args_and_arch(parser)

    main(ARGS)