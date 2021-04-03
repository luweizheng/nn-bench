import torch
import torch.optim as optim
import torch.nn as nn
import time
import argparse
from apex import amp
import sys
import os
import random

# sys.path.append(os.path.abspath("../../nns"))
# import nnutils
# import nnstats
from model import Encoder
from model import Decoder
from model import Seq2Seq

MAX = 2147483647
CLIP = 1

def get_synthetic_data(args):
    src = torch.randint(0, args.input_size, size=(args.fix_length, args.batch_size), dtype=torch.int64)
    trg = torch.randint(0, args.output_size, size=(args.fix_length, args.batch_size), dtype=torch.int64)
    return (src, trg)


# train iteration
def train(src, trg, model, criterion, optimizer, device, args):
    model.train()
    output = model(src, trg).to(device)
    # npu only accept int32 label
    if args.platform == "npu":
        trg = trg.to(torch.int32)

    output_dim = output.shape[-1]

    output = output[1:].view(-1, output_dim)
    trg = trg[1:].view(-1)

    loss = criterion(output, trg)
    
    if args.amp:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
    optimizer.step()

    return loss


def forward(src, trg, model, args):
    output = model(src, trg)
    return output

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.01)

def gen_seeds(num):
    return torch.randint(1, MAX, size=(num,), dtype=torch.float)

def main(args):
    
    if args.platform == "gpu":
        device = torch.device('cuda:' + args.device_id)
        device_func = torch.cuda
    elif args.platform == "npu":
        device = torch.device('npu:' + args.device_id)
        device_func = torch.npu
    else:
        device = torch.device('cpu')
    print("Running on device {}".format(device))

    (src, trg) = get_synthetic_data(args)
    src = src.to(device)
    trg = trg.to(device)

    seed_init = gen_seeds(32 * 1024 * 12).float().to(device)

    enc = Encoder(args.input_size, 
                args.embed_size, 
                args.hidden_size, 
                0.5, 
                platform=args.platform, 
                seed=seed_init).to(device)
    dec = Decoder(args.output_size, 
                args.embed_size, 
                args.hidden_size, 
                0.5, 
                platform=args.platform, 
                seed=seed_init).to(device)

    model = Seq2Seq(enc, dec, device).to(device)
    model.apply(init_weights)
    
    # define optimizer
    optimizer = optim.Adam(model.parameters())
    
    criterion = nn.CrossEntropyLoss().to(device)
    model = model.to(device)
    if args.amp:
        model, optimizer = amp.initialize(model, optimizer, 
                opt_level=args.opt_level, 
                loss_scale=args.loss_scale,
                verbosity=0)

    criterion = nn.CrossEntropyLoss().to(device)

    # warm up
    for i in range(args.num_warmups):
        if args.compute_type == "forward":
            predicted = forward(src, trg, model, args)
        elif args.compute_type == "backward":
            loss = train(src, trg, model, criterion, optimizer, device, args)
    device_func.synchronize()

    # bench
    # start time
    start_event = device_func.Event(enable_timing=True)
    end_event = device_func.Event(enable_timing=True)
    start_event.record()
    
    for i in range(args.num_iterations):
        if args.compute_type == "forward":
            predicted = forward(src, trg, model, args)
        elif args.compute_type == "backward":
            loss = train(src, trg, model, criterion, optimizer, device, args)
    
    # end time
    end_event.record()
    device_func.synchronize()
    elapsed_time = start_event.elapsed_time(end_event) / 1000
    example_per_sec = args.batch_size * args.num_iterations / elapsed_time
    # arithemetic_intensity = flop_sec / mem
    # flop_sec_scaled, flop_sec_unit = nnutils.unit_scale(flop_sec)
    
    print(f"time: {elapsed_time:.3f}")
    print(f"example per sec: {example_per_sec}")
    # print(f"flops_scaled: {flop_sec_scaled} {flop_sec_unit}")
    # print(f"arithemetic intensity: {arithemetic_intensity}")


    # 4. 执行forward+profiling
    # with torch.autograd.profiler.profile(**prof_kwargs) as prof:
    #     run()
    # print(prof.key_averages().table())
    # prof.export_chrome_trace("pytorch_prof_%s.prof" % args.device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Seq2Seq')
    parser.add_argument('--platform', type=str, default='npu',
                        help='set which type of device you want to use. gpu/npu')
    parser.add_argument('--device-id', type=str, default='0',
                        help='set device id')
    parser.add_argument('--amp', default=False, action='store_true',
                        help='use amp during prof')
    parser.add_argument('--loss-scale', default=64.0, type=float,
                        help='loss scale using in amp, default 64.0, -1 means dynamic')
    parser.add_argument('--opt-level', default='O2', type=str, help='opt-level using in amp, default O2')
    parser.add_argument('--dtype', type=str, default='float32', help='the data type')
    parser.add_argument('--num_iterations', type=int, default=100, help='the number of iterations')
    parser.add_argument('--num_warmups', type=int, default=10, help='number of warmup steps')
    parser.add_argument('--compute_type', type=str, default="forward", help='forward/backward')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--fix_length', type=int, default=64, help='fix length of the sentence')
    parser.add_argument('--embed_size', type=int, default=256, help='embedding size')
    parser.add_argument('--hidden_size', type=int, default=256, help='hidden size')
    parser.add_argument('--input_size', type=int, default=8192, help='input size')
    parser.add_argument('--output_size', type=int, default=8192, help='output size')

    args = parser.parse_args()

    main(args)