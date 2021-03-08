import os
import logging
import sys
logging.basicConfig(
    stream=sys.stdout, 
    format='{%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    level=logging.DEBUG)

import torch
from torch.nn import Module
from typing import Callable, Optional, Dict, Any, Tuple, List, Union, Iterable

from modules import module_mem, module_flops, module_macs
import nnutils
from nnutils import aggregate_info, format_info

__all__ = ['crawl_module', 'summary']


def apply(module: Module, fn: Callable[[Module, str], None], name: Optional[str] = None) -> None:
    """Modified version of `torch.nn.Module.apply` method

    Args:
        module: target module
        fn: function to apply to each module
        name: name of the current module
    """

    if name is None:
        name = module.__class__.__name__.lower()
    fn(module, name)
    for n, m in module.named_children():
        apply(m, fn, f"{name}.{n}")


def crawl_module(
    module: Module,
    input_shape: Union[List[Tuple[int, ...]], Tuple[int, ...]],
    dtype: Optional[Union[torch.dtype, Iterable[torch.dtype]]] = None
) -> Dict[str, Any]:
    """Retrieves module information for an expected input tensor shape

    Example::
        >>> import torch.nn as nn
        >>> from torchscan import summary
        >>> mod = nn.Conv2d(3, 8, 3)
        >>> module_info = crawl_module(mod, (3, 224, 224))

    Args:
        module: module to inspect
        input_shape: expected input shapes
        dtype: data type of each input argument to the module
    Returns:
        layer and overhead information
    """

    # Get device and data types from model
    p = next(module.parameters())
    device = p.device

    # input
    if not isinstance(input_shape, list):
        input_shape = [input_shape]
    if dtype is None:
        dtype = p.data.dtype
    if isinstance(dtype, torch.dtype):
        dtype = [dtype] * len(input_shape)
    # Tensor arguments
    input_ts = [torch.rand(in_shape).to(dtype=_dtype, device=device)
                for in_shape, _dtype in zip(input_shape, dtype)]

    pre_fw_handles, post_fw_handles = [], []
    pre_hook_tracker: Dict[int, Any] = {}
    post_hook_tracker: Dict[int, Any] = {}

    # Hook definition
    def _hook_info(module: Module, name: str) -> None:

        def _pre_hook(module: Module, input: torch.Tensor) -> None:
            """Pre-forward hook"""
            # Check that another hook has not been triggered at this forward stage
            if not pre_hook_tracker[id(module)]['is_used'] and \
               (pre_hook_tracker[id(module)]['target'] == pre_hook_tracker[id(module)]['current']):
                # Add information
                # Params
                grad_params, nograd_params, param_size = 0, 0, 0
                num_buffers, buffer_size = 0, 0
                is_shared = False
                if not any(module.children()):
                    # Parameters
                    for p in module.parameters():
                        if id(p) not in param_ids:
                            if p.requires_grad:
                                grad_params += p.data.numel()
                            else:
                                nograd_params += p.data.numel()
                            param_size += p.data.numel() * p.data.element_size()
                            param_ids.append(id(p))
                        else:
                            is_shared = True
                    # Buffers
                    for b in module.buffers():
                        if id(b) not in param_ids:
                            num_buffers += b.numel()
                            buffer_size += b.numel() * b.element_size()
                            param_ids.append(id(b))
                        else:
                            is_shared = True

                if call_idxs.get(id(module)) is None:
                    call_idxs[id(module)] = [len(info)]
                else:
                    call_idxs[id(module)].append(len(info))
                
                info.append(dict(name=name.rpartition('.')[-1],
                                 depth=len(name.split('.')) - 1,
                                 type=module.__class__.__name__,
                                 input_shape=(tuple(input[0].shape)),
                                 output_shape=None,
                                 grad_params=grad_params,
                                 nograd_params=nograd_params,
                                 param_size=param_size,
                                 num_buffers=num_buffers,
                                 buffer_size=buffer_size,
                                 flops=0,
                                 macs=0,
                                 mem=0,
                                 input_elements=0,
                                 output_elements=0,
                                 is_shared=is_shared,
                                 is_leaf=not any(module.children())))
                # Mark the next hook for execution
                pre_hook_tracker[id(module)]['target'] += 1
                # Current pass already used one of the hooks
                pre_hook_tracker[id(module)]['is_used'] = True
            pre_hook_tracker[id(module)]['current'] += 1
            # All the hooks have been checked, reset the temporary values
            if pre_hook_tracker[id(module)]['current'] == len(module._forward_pre_hooks):
                pre_hook_tracker[id(module)]['current'] = 0
                pre_hook_tracker[id(module)]['is_used'] = False

        def _fwd_hook(module: Module, input: torch.Tensor, output: torch.Tensor) -> None:
            """Post-forward hook"""

            # Check that another hook has not been triggered at this forward stage
            if not post_hook_tracker[id(module)]['is_used'] and \
               (post_hook_tracker[id(module)]['target'] == post_hook_tracker[id(module)]['current']):
                # Write information
                # Retrieve forward index
                if len(call_idxs[id(module)]) == 1:
                    fw_idx = call_idxs[id(module)][0]
                else:
                    # The first dictionary with output_shape=None is the correct one
                    for _idx in call_idxs[id(module)]:
                        if info[_idx]['output_shape'] is None:
                            fw_idx = _idx
                            break

                if any(module.children()):
                    tot_flops, tot_macs, tot_mem = 0, 0, 0
                else:
                    # Compute stats for standalone layers
                    tot_flops = module_flops(module, input[0], output)
                    tot_macs = module_macs(module, input[0], output)
                    tot_mem = module_mem(module, input[0], output)

                # Update layer information
                # RNN has two output tensor
                if isinstance(module, (torch.nn.RNN)):
                    info[fw_idx]['output_shape'] = (tuple(output[0].shape), tuple(output[1].shape))
                else:
                    info[fw_idx]['output_shape'] = tuple(output.shape)
                
                # Add them, since some modules can be used several times
                info[fw_idx]['flops'] = tot_flops
                info[fw_idx]['macs'] = tot_macs
                info[fw_idx]['mem'] = tot_mem

                # Mark the next hook for execution
                post_hook_tracker[id(module)]['target'] += 1
                # Current pass already used one of the hooks
                post_hook_tracker[id(module)]['is_used'] = True
            post_hook_tracker[id(module)]['current'] += 1
            # All the hooks have been checked, reset the temporary values
            if post_hook_tracker[id(module)]['current'] == len(module._forward_pre_hooks):
                post_hook_tracker[id(module)]['current'] = 0
                post_hook_tracker[id(module)]['is_used'] = False

        pre_fw_handles.append(module.register_forward_pre_hook(_pre_hook))
        post_fw_handles.append(module.register_forward_hook(_fwd_hook))
        # Handle modules that are used multiple times (with several hooks)
        pre_hook_tracker[id(module)] = dict(current=0, target=0, is_used=False)
        post_hook_tracker[id(module)] = dict(current=0, target=0, is_used=False)

    # Hook model
    info: List[Dict[str, Any]] = []
    param_ids: List[int] = []
    call_idxs: Dict[int, List[int]] = {}
    apply(module, _hook_info)

    # Forward
    with torch.no_grad():
        module(*input_ts)

    # Removes all hooks using their handles
    for handle in pre_fw_handles:
        handle.remove()
    for handle in post_fw_handles:
        handle.remove()

    grad_params, nograd_params, param_size = 0, 0, 0
    num_buffers, buffer_size = 0, 0
    for p in module.parameters():
        if p.requires_grad:
            grad_params += p.data.numel()
        else:
            nograd_params += p.data.numel()
        param_size += p.data.numel() * p.data.element_size()
    for b in module.buffers():
        num_buffers += b.numel()
        buffer_size += b.numel() * b.element_size()

    return dict(layers=info,
                overall=dict(grad_params=grad_params, nograd_params=nograd_params, param_size=param_size,
                             num_buffers=num_buffers, buffer_size=buffer_size))


def summary(
    module: Module,
    input_shape: Tuple[int, ...],
    wrap_mode: str = 'mid',
    max_depth: Optional[int] = None
) -> None:
    """Print module summary for an expected input tensor shape

    Example::
        >>> import torch.nn as nn
        >>> from torchscan import summary
        >>> mod = nn.Conv2d(3, 8, 3)
        >>> bs = 2
        >>> summary(mod, (bs, 3, 224, 224))

    Args:
        module: module to inspect
        input_shape: expected input shapes
        wrap_mode: if a value is too long, where the wrapping should be performed
        max_depth: maximum depth of layer information
    """

    # Get the summary dict
    module_info = crawl_module(module, input_shape)
    # Aggregate until max_depth
    if isinstance(max_depth, int):
        module_info = nnutils.aggregate_info(module_info, max_depth)
    # Format it and print it
    print(format_info(module_info, wrap_mode))

    return module_info

def get_flops_mem(
    module: Module,
    input_shape: Tuple[int, ...],
    max_depth: Optional[int] = None
):
    """Get module total flops and memory statistics for an expected input tensor shape

    Args:
        module: module to inspect
        input_shape: expected input shapes
        max_depth: maximum depth of layer information
    """

    # Get the summary dict
    module_info = crawl_module(module, input_shape)
    # Aggregate until max_depth
    # if isinstance(max_depth, int):
    #     module_info = nnutils.aggregate_info(module_info, max_depth)
    return nnutils.flops_mem(module_info)
