import warnings
from functools import reduce
from operator import mul

from torch import nn
from torch import Tensor
from torch.nn import Module
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd, _ConvTransposeNd
from torch.nn.modules.pooling import _MaxPoolNd, _AvgPoolNd, _AdaptiveMaxPoolNd, _AdaptiveAvgPoolNd
from typing import Union
import logging


__all__ = ['module_mem']


def module_mem(module: Module, input: Tensor, output: Tensor) -> int:
    """Estimate the number of memory accesses by the module.
    The implementation overhead is neglected

    Args:
        module (torch.nn.Module): PyTorch module
        input (torch.Tensor): input to the module
        output (torch.Tensor): output of the module
    Returns:
        int: number of mems
    """

    if isinstance(module, nn.Identity):
        return mem_identity(module, input, output)
    elif isinstance(module, nn.Flatten):
        return mem_flatten(module, input, output)
    elif isinstance(module, nn.Linear):
        return mem_linear(module, input, output)
    elif isinstance(module, (nn.ReLU, nn.ReLU6)):
        return mem_relu(module, input, output)
    elif isinstance(module, (nn.ELU, nn.LeakyReLU)):
        return mem_act_single_param(module, input, output)
    elif isinstance(module, nn.Sigmoid):
        return mem_sigmoid(module, input, output)
    elif isinstance(module, nn.Tanh):
        return mem_tanh(module, input, output)
    elif isinstance(module, _ConvTransposeNd):
        return mem_convtransposend(module, input, output)
    elif isinstance(module, _ConvNd):
        return mem_convnd(module, input, output)
    elif isinstance(module, _BatchNorm):
        return mem_bn(module, input, output)
    elif isinstance(module, (_MaxPoolNd, _AvgPoolNd)):
        return mem_pool(module, input, output)
    elif isinstance(module, (_AdaptiveMaxPoolNd, _AdaptiveAvgPoolNd)):
        return mem_adaptive_pool(module, input, output)
    elif isinstance(module, nn.Dropout):
        return mem_dropout(module, input, output)
    else:
        warnings.warn(f'Module type not supported: {module.__class__.__name__}')
        return 0


def num_params(module: Module) -> int:
    """Compute the number of parameters

    Args:
        module (torch.nn.Module): PyTorch module
    Returns:
        int: number of parameter elements
    """

    return sum(p.data.numel() for p in module.parameters())


def mem_identity(module: nn.Identity, input: Tensor, output: Tensor) -> int:
    """MEMs estimation for `torch.nn.Identity`"""

    return input.numel()


def mem_flatten(module: nn.Flatten, input: Tensor, output: Tensor) -> int:
    """MEMs estimation for `torch.nn.Flatten`"""

    return 2 * input.numel()


def mem_linear(module: nn.Linear, input: Tensor, output: Tensor) -> int:
    """MEMs estimation for `torch.nn.Linear`"""

    input_dma = input.numel()
    # Access weight and bias
    ops_dma = num_params(module)
    output_dma = output.numel()

    return input_dma + ops_dma + output_dma


def mem_relu(module: Union[nn.ReLU, nn.ReLU6], input: Tensor, output: Tensor) -> int:
    """MEMs estimation for `torch.nn.ReLU`"""

    input_dma = input.numel()
    output_dma = 0 if module.inplace else output.numel()

    return input_dma + output_dma


def mem_act_single_param(module: Union[nn.ELU, nn.LeakyReLU], input: Tensor, output: Tensor) -> int:
    """MEMs estimation for activations with single parameter"""

    input_dma = input.numel()
    # Access alpha, slope or other
    ops_dma = 1
    output_dma = 0 if module.inplace else output.numel()

    return input_dma + ops_dma + output_dma


def mem_sigmoid(module: nn.Sigmoid, input: Tensor, output: Tensor) -> int:
    """MEMs estimation for `torch.nn.Sigmoid`"""

    # Access for both exp
    input_dma = input.numel()
    output_dma = output.numel()

    return input_dma + output_dma


def mem_tanh(module: nn.Tanh, input: Tensor, output: Tensor) -> int:
    """MEMs estimation for `torch.nn.Tanh`"""

    # Access for both exp
    input_dma = input.numel() * 2
    output_dma = output.numel()

    return input_dma + output_dma


def mem_dropout(module: nn.Dropout, input: Tensor, output: Tensor) -> int:
    """MEMs estimation for `torch.nn.Dropout`"""

    input_dma = input.numel()

    # Access sampling probability
    ops_dma = 1

    output_dma = 0 if module.inplace else output.numel()

    return input_dma + ops_dma + output_dma


def mem_convtransposend(module: _ConvTransposeNd, input: Tensor, output: Tensor) -> int:
    """MEMs estimation for `torch.nn.modules.conv._ConvTransposeNd`"""

    # Padding (# cf. https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py#L496-L532)
    # Access stride, padding and kernel_size
    in_padding = len(module.kernel_size) * 4
    out_padding = len(module.kernel_size)

    # The rest is like a classic convolution
    conv_MEMs = mem_convnd(module, input, output)

    return in_padding + out_padding + conv_MEMs


def mem_convnd(module: _ConvNd, input: Tensor, output: Tensor) -> int:
    """MEMs estimation for `torch.nn.modules.conv._ConvNd`"""

    # Each output element required K ** 2 memory access of each input channel

    input_dma = input.numel()

    # Access weight & bias
    ops_dma = num_params(module)
    output_dma = output.numel()

    return input_dma + ops_dma + output_dma


def mem_bn(module: _BatchNorm, input: Tensor, output: Tensor) -> int:
    """MEMs estimation for `torch.nn.modules.batchnorm._BatchNorm`"""
    input_dma = input.numel()

    # Access running_mean, running_var and eps
    ops_dma = module.running_mean.numel() + module.running_var.numel() + 1  # type: ignore[operator]
    # Access to weight and bias
    if module.affine:
        ops_dma += module.weight.data.numel() + module.bias.data.numel()
    # Exp avg factor
    if module.momentum:
        ops_dma += 1
    # Update stats
    if module.training and module.track_running_stats:
        # Current mean and std computation only requires access to input, already counted in input_dma
        # Update num of batches and running stats
        ops_dma += 1 + module.running_mean.numel() + module.running_var.numel()  # type: ignore[operator]

    output_dma = output.numel()

    return input_dma + ops_dma + output_dma


def mem_pool(module: Union[_MaxPoolNd, _AvgPoolNd], input: Tensor, output: Tensor) -> int:
    """MEMs estimation for spatial pooling modules"""

    # Resolve kernel size and stride size (can be stored as a single integer or a tuple)
    if isinstance(module.kernel_size, tuple):
        kernel_size = module.kernel_size
    else:
        kernel_size = (module.kernel_size,) * (input.ndim - 2)  # type: ignore[attr-defined]

    # Each output element required K ** 2 memory accesses
    input_dma = reduce(mul, kernel_size) * output.numel()

    output_dma = output.numel()

    return input_dma + output_dma


def mem_adaptive_pool(module: Union[_AdaptiveMaxPoolNd, _AdaptiveAvgPoolNd], input: Tensor, output: Tensor) -> int:
    """MEMs estimation for adaptive spatial pooling modules"""

    if isinstance(module.output_size, tuple):
        o_sizes = module.output_size
    else:
        o_sizes = (module.output_size,) * (input.ndim - 2)  # type: ignore[attr-defined]
    # Approximate kernel_size using ratio of spatial shapes between input and output
    kernel_size = tuple(i_size // o_size if (i_size % o_size) == 0 else i_size - o_size * (i_size // o_size) + 1
                        for i_size, o_size in zip(input.shape[2:], o_sizes))
    # Each output element required K ** 2 memory accesses
    input_dma = reduce(mul, kernel_size) * output.numel()

    output_dma = output.numel()

    return input_dma + output_dma


def mem_rnn(module: nn.RNN, input: Tensor, output: Tensor) -> dict:
    """number of elements estimation for `torch.nn.RNN`"""

    if module.batch_first == True:
        batch_size = input.shape[0]
        seq_length = input.shape[1]
    else:
        batch_size = input.shape[1]
        seq_length = input.shape[0]

    # input tensor, include input X and hidden
    input_numel = input.numel()

    logging.debug(f"input tensor shape {input.shape}, input elements {input.numel()}")
    # Access weight and bias
    params = params_size(module)

    logging.debug(f"params {params}")

    # output is a tuple () tensor 
    logging.debug(f"output tensor shape {output[0].shape}, hidden shape {output[1].shape}")
    output_numel = output[0].numel() + output[1].numel()

    return dict(input=input_numel, params=params, output=output_numel)
