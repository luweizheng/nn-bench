import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import argparse
import time
import sys
import logging

sys.path.append(os.path.abspath("../../nns"))
import nnstats
import nnutils

# calibration measurement
def run_calibrate(input_tensor, model):
    output_result = input_tensor

# forward
def run_forward(input_tensor, model):
    output_result = model(input_tensor)

# backward
def run_backward(input_tensor, model):
    output_result = model(input_tensor)

    lr = 0.01
    momentum = 0.5
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    optimizer.zero_grad()

    loss = torch.sum(output_result)
    loss.backward()
    optimizer.step()


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

    # select commpute type
    if args.compute_type == "forward":
        compfunc = run_forward
    elif args.compute_type == "backward":
        compfunc = run_backward
    elif args.compute_type == "calibrate":
        compfunc = run_calibrate
    else:
        raise ValueError(
            "Error, compute_type should be either forward or backward or calibrate")

    kernel_shape = args.kernel_shape
    # requires_grad=True indicates that we want to compute gradients during the backward pass
    if args.compute_type == "backward":
        weights = torch.randn(kernel_shape[3], kernel_shape[2], kernel_shape[0],
                              kernel_shape[1], device=device, requires_grad=True)
        biases = torch.randn(
            kernel_shape[3], device=device, requires_grad=True)
    else:
        weights = torch.randn(kernel_shape[3], kernel_shape[2], kernel_shape[0],
                              kernel_shape[1], device=device)
        biases = torch.randn(kernel_shape[3], device=device)

    input_tensor_shape = tuple(args.input_tensor_shape)
    input_tensor = torch.randn(input_tensor_shape[0], input_tensor_shape[1],
                              input_tensor_shape[2], input_tensor_shape[3], device=device)
    
    if args.dtype == "float16":
        input_tensor = input_tensor.half()

    # init the conv2d kernel
    conv2d = nn.Conv2d(in_channels=kernel_shape[2], out_channels=kernel_shape[3], kernel_size=kernel_shape[0], stride=args.stride)
    conv2d.weight = torch.nn.Parameter(weights)
    conv2d.bias = torch.nn.Parameter(biases)
    # move the kernel to device
    conv2d.to(device)
    if args.dtype == "float16":
        conv2d = conv2d.half()

    # warm up
    conv2d.eval()
    flops, mem = nnstats.get_flops_mem(conv2d, input_tensor_shape)

    if args.dtype == 'float16':
        mem = mem * 2
    elif args.dtype == 'float32':
        mem = mem * 4

    if args.compute_type == "forward":
        flops = flops
    elif args.compute_type == "backward":
        flops = flops * 3
    else:
        flop_sec = 0.0
    for i in range(args.num_warmups):
        compfunc(input_tensor, conv2d)
    device_func.synchronize()
    
    # bench
    start_event = device_func.Event(enable_timing=True)
    end_event = device_func.Event(enable_timing=True)
    start_event.record()

    for i in range(args.num_iterations):
        compfunc(input_tensor, conv2d)

    end_event.record()
    device_func.synchronize()
    elapsed_time = start_event.elapsed_time(end_event) / 1000

    flop_sec = flops * args.num_iterations / elapsed_time
    example_per_sec = input_tensor_shape[0] * args.num_iterations / elapsed_time
    flop_sec_scaled, flop_sec_unit = nnutils.unit_scale(flop_sec)
    mem_scaled, mem_unit = nnutils.unit_scale(mem)

    print(f"-----performance----")
    print(f"\n")
    print(f"device time: {elapsed_time:.6f}")
    print(f"flops: {flop_sec}")
    print(f"memory: {mem}")
    print(f"example_per_sec: {example_per_sec:.3f}")
    print(f"flops_scaled: {flop_sec_scaled} {flop_sec_unit}")
    print(f"memory_scaled: {mem_scaled} {mem_unit}")


if __name__ == '__main__':
    AP = argparse.ArgumentParser()
    AP.add_argument('--platform', type=str, default="npu",
                    help='neural accelerator platform, cuda or npu')
    AP.add_argument('--input_tensor_shape', type=int, nargs='+', default=[64, 3, 224, 224],
                    help='the shape of the input tensor. Note that it depends on data_format (default NCHW)')
    AP.add_argument('--kernel_shape', type=int, nargs='+', default=[
                    3, 3, 3, 64], help='the shape of the conv kernel [filter_height, filter_width, in_channels, out_channels]')
    AP.add_argument('--stride', type=int, default=1, help='the stride')
    AP.add_argument('--dtype', type=str, default='float32',
                    help='the data type')
    AP.add_argument('--num_iterations', type=int, default=100,
                    help='the number of iterations')
    AP.add_argument('--num_warmups', type=int, default=10,
                    help='number of warmup steps')
    AP.add_argument('--compute_type', type=str,
                    default="forward", help='forward or backward pass')
    args = AP.parse_args()

    # print args
    # for arg in vars(args):
    #     print(arg, ":", getattr(args, arg))

    main(args=args)
