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


def run_calibrate(input_image, func):
    output_result = input_image

# forward


def run_forward(input_image, func):
    output_result = func(input_image)

# backward


def run_backward(input_image, func):
    output_result = func(input_image)

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
        weights = torch.randn(kernel_shape[3], kernel_shape[2], kernel_shape[0],
                              kernel_shape[1], device=device, dtype=tensor_type, requires_grad=True)
        biases = torch.randn(
            kernel_shape[3], device=device, dtype=tensor_type, requires_grad=True)
    else:
        weights = torch.randn(kernel_shape[3], kernel_shape[2], kernel_shape[0],
                              kernel_shape[1], device=device, dtype=tensor_type)
        biases = torch.randn(kernel_shape[3], device=device, dtype=tensor_type)

    input_tensor_shape = args.input_tensor_shape
    # the input format is NHWC, pytorch requires NCHW thus we do a transpose here
    input_image = torch.randn(input_tensor_shape[0], input_tensor_shape[1],
                              input_tensor_shape[2], input_tensor_shape[3], device=device, dtype=tensor_type)

    # init the conv2d kernel
    conv2d = nn.Conv2d(
        in_channels=kernel_shape[2], out_channels=kernel_shape[3], kernel_size=kernel_shape[0], stride=args.stride)
    conv2d.weight = torch.nn.Parameter(weights)
    conv2d.bias = torch.nn.Parameter(biases)
    # move the kernel to device
    conv2d.to(device)

    # start session
    print("warming up for {} steps".format(args.num_warmups))
    start = time.time()
    conv2d.eval()
    flops, mem = nnstats.get_flops_mem(conv2d, (64, 3, 224, 224))
    if args.compute_type == "forward":
        flops = flops
    elif args.compute_type == "backward":
        flops = flops * 3
    else:
        flop_sec = 0.0
    print(f"{flops}, {mem}")
    for i in range(args.num_warmups):
        compfunc(input_image, conv2d)
    end = time.time()
    # logging.debug(f"Max memory used by tensors = {device_func.max_memory_allocated()} bytes")
    # logging.debug(f"Max memory used by tensors = {device_func.memory_allocated()} bytes")
    
    # device_func.reset_max_memory_allocated()
    device_func.synchronize()
    print("done")
    duration = end - start
    print('Warmup {:.2f} seconds, {:.2f} seconds/iter'.format(duration,
                                                              duration/float(args.num_warmups)))

    print("running for {} steps".format(args.num_iterations))
    start = time.time()
    start_event = device_func.Event(enable_timing=True)
    end_event = device_func.Event(enable_timing=True)
    start_event.record()

    for i in range(args.num_iterations):
        compfunc(input_image, conv2d)

    end_event.record()
    device_func.synchronize()
    # max_mem = device_func.max_memory_allocated()
    
    # logging.debug(f"Max memory used by tensors = {device_func.memory_allocated()} bytes")
    elapsed_time = start_event.elapsed_time(end_event) / 1000
    end = time.time()
    print("done")

    duration = end - start
    flop_sec = flops * args.num_iterations / elapsed_time
    flop_sec_scaled, flop_sec_unit = nnutils.unit_scale(flop_sec)
    mem_scaled, mem_unit = nnutils.unit_scale(mem)
    # max_mem_scaled, max_mem_unit = nnutils.unit_scale(max_mem)

    print(f"time.time {duration:.6f} seconds device.time {elapsed_time:.6f}")
    print(f"FLOPS: {flop_sec}")
    print(f"memory: {mem}")

    print(f"scale_flops: {flop_sec_scaled} {flop_sec_unit}")
    print(f"scale_mem: {mem_scaled} {mem_unit}")
    # print(f"max_scale_mem: {max_mem_scaled} {max_mem_unit}")


if __name__ == '__main__':
    AP = argparse.ArgumentParser()
    AP.add_argument('--platform', type=str, default="npu",
                    help='neural accelerator platform, cuda or npu')
    AP.add_argument('--input_tensor_shape', type=int, nargs='+', default=[64, 3, 224, 224],
                    help='the shape of the input tensor. Note that it depends on data_format (default NHWC)')
    AP.add_argument('--data_format', type=str, default='NHWC',
                    help='choose either channels_last or channels_first')
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
