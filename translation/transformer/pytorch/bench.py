import torch
import torch.optim as optim
import time
from apex import amp

import sys
from utils import options, utils, criterions
import data
from data import data_utils
from models import build_model
import numpy as np

MAX = 2147483647
def _gen_seeds(shape):
    return np.random.uniform(1, MAX, size=shape).astype(np.float32)
seed_shape = (32 * 1024 * 12, )


def main(args):
    # print(args)

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

    # genernate data
    src_dict, tgt_dict = data_utils.load_dictionaries(args)
    datasets = data_utils.get_dummy_batch(args.max_sentences, src_dict, tgt_dict)
    assert len(datasets) > 0, "empty genernated data!"
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

    # build model
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

    # warm up
    for i in range(10):
        train(args, model, src_tokens, src_lengths, prev_output_tokens, target, criterion, optimizer)
    
    # bench
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