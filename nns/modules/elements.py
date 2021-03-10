import warnings
import logging
from functools import reduce
from operator import mul

from torch import nn
from torch import Tensor
from torch.nn import Module
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd, _ConvTransposeNd
from torch.nn.modules.pooling import _MaxPoolNd, _AvgPoolNd, _AdaptiveMaxPoolNd, _AdaptiveAvgPoolNd
from typing import Union

__all__ = ['module_numel']

def num_params(module: Module) -> int:
    """Compute the number of parameters

    Args:
        module (torch.nn.Module): PyTorch module
    Returns:
        int: number of parameter elements
    """

    return sum(p.data.numel() for p in module.parameters())

def module_numel(module: Module, input: Tensor, output: Tensor) -> dict():
    """Estimate the number of element of the module, including input elements, output elements, weights and bias.

    Args:
        module (torch.nn.Module): PyTorch module
        input (torch.Tensor): input to the module
        output (torch.Tensor): output of the module
    Returns:
        dict: (input -> , params -> , output -> )
    """

    if isinstance(module, nn.Identity):
        return numel_identity(module, input, output)
    elif isinstance(module, nn.Flatten):
        return numel_flatten(module, input, output)
    elif isinstance(module, nn.Linear):
        return numel_linear(module, input, output)
    elif isinstance(module, (nn.ReLU, nn.ReLU6)):
        return numel_relu(module, input, output)
    elif isinstance(module, (nn.ELU, nn.LeakyReLU)):
        return numel_act_single_param(module, input, output)
    elif isinstance(module, nn.Sigmoid):
        return numel_sigmoid(module, input, output)
    elif isinstance(module, nn.Tanh):
        return numel_tanh(module, input, output)
    elif isinstance(module, _ConvTransposeNd):
        return numel_convtransposend(module, input, output)
    elif isinstance(module, _ConvNd):
        return numel_convnd(module, input, output)
    elif isinstance(module, _BatchNorm):
        return numel_bn(module, input, output)
    elif isinstance(module, (_MaxPoolNd, _AvgPoolNd)):
        return numel_pool(module, input, output)
    elif isinstance(module, (_AdaptiveMaxPoolNd, _AdaptiveAvgPoolNd)):
        return numel_adaptive_pool(module, input, output)
    elif isinstance(module, nn.Dropout):
        return numel_dropout(module, input, output)
    elif isinstance(module, nn.RNN):
        return numel_rnn(module, input, output)
    else:
        warnings.warn(f'Module type not supported: {module.__class__.__name__}')
        return dict(input=0, params=0, output=0)


def numel_convnd(module: _ConvNd, input: Tensor, output: Tensor) -> dict:
    """number of elements estimation for `_ConvNd`"""

    input_numel = input.numel()
    # Access weight and bias
    params = num_params(module)
    output_numel = output.numel()

    return dict(input=input_numel, params=params, output=output_numel)

def numel_linear(module: nn.Linear, input: Tensor, output: Tensor) -> dict:
    """number of elements estimation for `torch.nn.Linear`"""

    input_numel = input.numel()
    # Access weight and bias
    params = num_params(module)
    output_numel = output.numel()

    return dict(input=input_numel, params=params, output=output_numel)

def numel_rnn(module: nn.RNN, input: Tensor, output: Tensor) -> dict:
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
    params = num_params(module)

    logging.debug(f"params {params}")

    # output is a tuple () tensor 
    logging.debug(f"output tensor shape {output[0].shape}, hidden shape {output[1].shape}")
    output_numel = output[0].numel() + output[1].numel()

    return dict(input=input_numel, params=params, output=output_numel)