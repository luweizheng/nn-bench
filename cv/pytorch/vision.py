import torch
import torch.optim as optim
import torch.nn as nn
import time
import argparse
import ssl
import torchvision
from apex import amp
import sys
import os
import random

sys.path.append(os.path.abspath("../../nns"))
import nnutils
import nnstats

ssl._create_default_https_context = ssl._create_unverified_context

def get_synthetic_data(input_shape):
    input_tensor = torch.randn(input_shape)
    label = torch.randint(0, 1000, (input_shape[0],))
    return (input_tensor, label)


# train iteration
def train(input_tensor, label, model, criterion, optimizer, args):
    # npu only accept int32 label
    if args.platform == "npu":
        label = label.to(torch.int32)
    output_tensor = model(input_tensor)
    loss = criterion(output_tensor, label)
    optimizer.zero_grad()
    if args.amp:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
    optimizer.step()
    return loss


def forward(input_tensor, model, args):
    output_tensor = model(input_tensor)
    return output_tensor

def main(args):
    
    if args.platform == "gpu":
        device = torch.device('cuda:0')
        device_func = torch.cuda
    elif args.platform == "npu":
        device = torch.device('npu:0')
        device_func = torch.npu
    else:
        device = torch.device('cpu')
    print("Running on device {}".format(device))

    input_tensor_shape = tuple(args.input_tensor_shape)
    (input_tensor, label) = get_synthetic_data(input_tensor_shape)
    input_tensor = input_tensor.to(device)
    label = label.to(device)

    model = torchvision.models.__dict__[args.arch](args.pretrained)
    # model.eval()
    flops, mem = nnstats.get_flops_mem(model, input_tensor_shape)
    
    if args.dtype == 'float16':
        model = model.half()
        input_tensor = input_tensor.half()
        mem = mem * 2
    elif args.dtype == 'float32':
        mem = mem * 4

    if args.compute_type == "forward":
        flops = flops
    elif args.compute_type == "backward":
        flops = flops * 3
    else:
        flop_sec = 0.0

    
    # define optimizer
    if args.FusedSGD:
        from apex.optimizers import NpuFusedSGD
        optimizer = NpuFusedSGD(model.parameters(), lr=0.01)
        model = model.to(device)
        if args.amp:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level,
                                              loss_scale=None if args.loss_scale == -1 else args.loss_scale,
                                              combine_grad=False)
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        model = model.to(device)
        if args.amp:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level,
                                              loss_scale=None if args.loss_scale == -1 else args.loss_scale)


    criterion = nn.CrossEntropyLoss().to(device)


    # warm up
    for i in range(args.num_warmups):
        if args.compute_type == "forward":
            predicted = forward(input_tensor, model, args)
        elif args.compute_type == "backward":
            loss = train(input_tensor, label, model, criterion, optimizer, args)
    device_func.synchronize()

    # bench
    # start time
    start_event = device_func.Event(enable_timing=True)
    end_event = device_func.Event(enable_timing=True)
    start_event.record()
    
    for i in range(args.num_iterations):
        if args.compute_type == "forward":
            predicted = forward(input_tensor, model, args)
        elif args.compute_type == "backward":
            loss = train(input_tensor, label, model, criterion, optimizer, args)
    
    # end time
    end_event.record()
    device_func.synchronize()
    elapsed_time = start_event.elapsed_time(end_event) / 1000
    flop_sec = flops * args.num_iterations / elapsed_time
    arithemetic_intensity = flop_sec / mem
    flop_sec_scaled, flop_sec_unit = nnutils.unit_scale(flop_sec)
    
    print(f"flops: {flop_sec}")
    print(f"time: {elapsed_time:.3f}")
    print(f"flops_scaled: {flop_sec_scaled} {flop_sec_unit}")
    print(f"arithemetic intensity: {arithemetic_intensity}")


    # 4. 执行forward+profiling
    # with torch.autograd.profiler.profile(**prof_kwargs) as prof:
    #     run()
    # print(prof.key_averages().table())
    # prof.export_chrome_trace("pytorch_prof_%s.prof" % args.device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ResNet')
    parser.add_argument('--platform', type=str, default='gpu',
                        help='set which type of device you want to use. gpu/npu')
    parser.add_argument('--amp', default=False, action='store_true',
                        help='use amp during prof')
    parser.add_argument('--loss-scale', default=64.0, type=float,
                        help='loss scale using in amp, default 64.0, -1 means dynamic')
    parser.add_argument('--opt-level', default='O2', type=str,
                        help='opt-level using in amp, default O2')
    parser.add_argument('--FusedSGD', default=False, action='store_true',
                        help='use FusedSGD during prof')
    parser.add_argument('--dtype', type=str, default='float32',
                    help='the data type')
    parser.add_argument('--num_iterations', type=int, default=100,
                    help='the number of iterations')
    parser.add_argument('--num_warmups', type=int, default=10,
                    help='number of warmup steps')
    parser.add_argument('--compute_type', type=str,
                    default="forward", help='forward/backward')
    parser.add_argument('--arch', type=str,
                    default="resnet50", help='model architecture')
    parser.add_argument('--pretrained', type=bool,
                    default="", help='use pre-trained model or not')

    parser.add_argument('--input_tensor_shape', type=int, nargs='+', default=[64, 3, 224, 224],
                    help='the shape of the input tensor. Note that it depends on data_format (default NHWC)')
    args = parser.parse_args()

    main(args)