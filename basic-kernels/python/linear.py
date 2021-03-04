import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import cupy
import argparse
import time
import sys

sys.path.append(os.path.abspath("../../nns"))
import nnutils
import nnstats

# calibration measurement
def run_calibrate(input_tensor, func):
    output_result = input_tensor

# forward
def run_forward(input_tensor, func):
    output_result = func(input_tensor)

# backward
def run_backward(input_tensor, func):
    output_result = func(input_tensor)

    #lr = 0.01
    #momentum = 0.5
    #optimizer = optim.SGD(conv2d.parameters(), lr=lr, momentum=momentum)
    # optimizer.zero_grad()

    res = torch.sum(output_result)
    res.backward()
    # optimizer.step()


def main(args):

    # datatype selection
    if args.dtype == 'float16':
        tensor_type = torch.float16
    elif args.dtype == 'float32':
        tensor_type = torch.float32
    else:
        raise Exception('data type can only be float16 or float32')

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
        weights = torch.randn(kernel_shape[1], kernel_shape[0], device=device, dtype=tensor_type, requires_grad=True)
        biases = torch.randn(kernel_shape[1], device=device, dtype=tensor_type, requires_grad=True)
    else:
        weights = torch.randn(kernel_shape[1], kernel_shape[0], device=device, dtype=tensor_type)
        biases = torch.randn(kernel_shape[1], device=device, dtype=tensor_type)

    input_tensor_shape = args.input_tensor_shape
    # the input format is NHWC, pytorch requires NCHW thus we do a transpose here
    input_tensor = torch.randn(input_tensor_shape[0], input_tensor_shape[1]
                              , device=device, dtype=tensor_type)

    # init the Linear kernel
    linear = nn.Linear(in_features=kernel_shape[0], out_features=kernel_shape[1]).eval()
    linear.weight = torch.nn.Parameter(weights)
    linear.bias = torch.nn.Parameter(biases)
    # move the kernel to device
    linear.to(device)

    # start session
    print("warming up for {} steps".format(args.num_warmups))
    start = time.time()
    linear.eval()
    flops, mem = nnstats.get_flops_mem(linear, input_tensor_shape)
    if args.compute_type == "forward":
        flops = flops
    elif args.compute_type == "backward":
        flops = flops * 3
    else:
        flop_sec = 0.0
    print(f"{nnutils.unit_scale(flops)}, {nnutils.unit_scale(mem)}")
    for i in range(args.num_warmups):
        compfunc(input_tensor, linear)
    end = time.time()
    print("done")
    duration = end - start
    print('Warmup {:.2f} seconds, {:.2f} seconds/iter'.format(duration,
                                                              duration/float(args.num_warmups)))

    print("running for {} steps".format(args.num_iterations))
    start = time.time()
    start_event = device_func.Event(enable_timing=True)
    end_event = device_func.Event(enable_timing=True)
    start_event.record()

    # cupy.cuda.profiler.start()
    for i in range(args.num_iterations):
        compfunc(input_tensor, linear)

    end_event.record()
    device_func.synchronize()  # Wait for the events to be recorded!
    elapsed_time = start_event.elapsed_time(end_event) / 1000
    end = time.time()
    print("done")

    flop_sec = flops * args.num_iterations / elapsed_time

    duration = end - start
    flop_sec_scaled, flop_sec_unit = nnutils.unit_scale(flop_sec)
    mem_scaled, mem_unit = nnutils.unit_scale(mem)
    
    print(f"time.time {duration:.6f} seconds cuda.time {elapsed_time:.6f}")
    print(f"FLOPS: {flop_sec_scaled:.6f} {flop_sec_unit}, memory access: {mem_scaled:.6f} {mem_unit}")



if __name__ == '__main__':
    AP = argparse.ArgumentParser()
    AP.add_argument('--platform', type=str, default="npu",
                    help='neural accelerator platform, cuda or npu')
    AP.add_argument('--input_tensor_shape', type=int, nargs='+', default=(256, 16384), 
                    help='the shape of the input tensor. usually (N, Hin)')
    AP.add_argument('--kernel_shape', type=int, nargs='+', default=(16384, 16384), 
                    help='the shape of the linear kernel [in_features, out_features]')
    AP.add_argument('--dtype', type=str, default='float32',
                    help='the data type')
    AP.add_argument('--num_iterations', type=int, default=100,
                    help='the number of iterations')
    AP.add_argument('--num_warmups', type=int, default=10,
                    help='number of warmup steps')
    AP.add_argument('--compute_type', type=str,
                    default="forward", help='forward/backward')
    args = AP.parse_args()

    # print args
    # for arg in vars(args):
    #     print(arg, ":", getattr(args, arg))

    main(args=args)