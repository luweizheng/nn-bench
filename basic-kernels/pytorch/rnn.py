# A rnn1d kernel using pytorch

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

import logging
import sys

sys.path.append(os.path.abspath("../../nns"))
import nnutils
import nnstats


def run_calibrate(input_tensor, myRNN):
    output_result = input_tensor

def run_forward(input_tensor, myRNN):
    output_result = myRNN(input_tensor)
    
def run_backward(input_tensor, myRNN):
    lr = 0.01
    momentum = 0.5
    optimizer = optim.SGD(myRNN.parameters(), lr=lr, momentum=momentum)
    optimizer.zero_grad()
        
    output_result, _ = myRNN(input_tensor)
    output_result.sum().backward()
    optimizer.step()
    
def main(args): 

    if args.dtype == 'float16':
        tensor_type=torch.float16
    elif args.dtype == 'float32':
        tensor_type=torch.float32
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
    
    input_tensor_shape = tuple(args.input_tensor_shape)
    # the input format is (seq_len, batch, input_size)
    input_tensor = torch.randn(input_tensor_shape[0], input_tensor_shape[1], input_tensor_shape[2], device=device, dtype=tensor_type)

    input_size = input_tensor_shape[2]
    hidden_size = args.n_neurons

    # init rnn kernel
    if args.cell_type == 'lstm':
        myRNN = nn.LSTM(input_size, hidden_size, batch_first=True)
    elif args.cell_type == 'rnn':
        myRNN = nn.RNN(input_size, hidden_size)
    elif args.cell_type == 'gru':
        myRNN = nn.GRU(input_size, hidden_size, batch_first=True)
    else:
        raise ValueError("Error of input cell_type, please choose one from [rnn, lstm, gru]")
    if (tensor_type == torch.float16):
        myRNN = myRNN.half()
    # move the kernel to GPU
    myRNN.to(device)
   
    # resul ops
    if args.compute_type=="forward":
        compfunc = run_forward  
    elif args.compute_type=="backward":
        compfunc = run_backward
    elif args.compute_type=="calibrate":
        compfunc = run_calibrate
    else:
        raise ValueError("Error, compute_type should be either forward or backward or calibrate")
    
    
    print("warming up for {} steps".format(args.num_warmups))
    start = time.time()

    flops, mem = nnstats.get_flops_mem(myRNN, input_tensor_shape)
    for i in range(args.num_warmups):
        compfunc(input_tensor, myRNN)
    device_func.synchronize()
    end = time.time()
    print("done")
    duration = end-start

    logging.debug(f"Max memory used by tensors = {torch.cuda.max_memory_allocated()} bytes")
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    print('Warmup {:.2f} seconds, {:.2f} seconds/iter'.format(duration, duration/float(args.num_warmups)))
    
    print("running for {} steps".format(args.num_iterations))
    start = time.time()
    start_event = device_func.Event(enable_timing=True)
    end_event = device_func.Event(enable_timing=True)
    start_event.record()
            
    for i in range(args.num_iterations):
        compfunc(input_tensor, myRNN)
        
    end_event.record()
    device_func.synchronize()  # Wait for the events to be recorded!
    logging.debug(f"Max memory used by tensors = {torch.cuda.max_memory_allocated()} bytes")
    end = time.time()
    print("done")
        
    end = time.time()
    elapsed_time = start_event.elapsed_time(end_event) / 1000
    
    # print("done")

    flop_sec = flops * args.num_iterations / elapsed_time

    duration = end - start
    flop_sec_scaled, flop_sec_unit = nnutils.unit_scale(flop_sec)
    mem_scaled, mem_unit = nnutils.unit_scale(mem)
    
    print(f"time.time {duration:.6f} seconds device.time {elapsed_time:.6f}")
    print(f"FLOPS: {flop_sec}")
    print(f"memory: {mem}")



if __name__ == '__main__':
    AP = argparse.ArgumentParser()
    AP.add_argument('--platform', type=str, default="gpu",
                    help='neural accelerator platform, cuda or npu')
    AP.add_argument('--input_tensor_shape', type=int, nargs='+', default=[16, 2, 10], 
                    help='the shape of the input tensor, shape (seq_len, batch_size, input_size)')
    AP.add_argument('--cell_type', type=str, default='rnn', help='the rnn cell type')
    AP.add_argument('--n_neurons', type=int, default=20, help='number of neurons for the layer, or hidden size')
    AP.add_argument('--dtype', type=str, default='float32', help='the data type')
    AP.add_argument('--num_iterations', type=int, default=100, help='the number of iterations')
    AP.add_argument('--num_warmups', type=int, default=10, help='number of warmup steps')
    AP.add_argument('--compute_type', type=str, default="forward", help='forward or backward pass')
    args = AP.parse_args()
    
    #print args
    # for arg in vars(args):
    #     print(arg, ":", getattr(args, arg))
        
    
    main(args)
    

