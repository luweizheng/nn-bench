from typing import Tuple, Dict, Any, Optional, List


def format_name(name: str, depth: int = 0) -> str:
    """Format a string for nested data printing

    Args:
        name: input string
        depth: depth of the nested information
    Returns:
        formatted string
    """

    if depth == 0:
        return name
    elif depth == 1:
        return f"├─{name}"
    else:
        return f"{'|    ' * (depth - 1)}└─{name}"


def wrap_string(s: str, max_len: int, delimiter: str = '.', wrap: str = '[...]', mode: str = 'end') -> str:
    """Wrap a string into a given length

    Args:
        s: input string
        max_len: maximum string length
        delimiter: character used for delimiting information categories
        wrap: wrapping sequence used
        mode: wrapping mode
    Returns:
        wrapped string
    """

    if len(s) <= max_len or mode is None:
        return s

    if mode == 'end':
        return s[:max_len - len(wrap)] + wrap
    elif mode == 'mid':
        final_part = s.rpartition(delimiter)[-1]
        wrapped_end = f"{wrap}.{final_part}"
        return s[:max_len - len(wrapped_end)] + wrapped_end
    else:
        raise ValueError("received an unexpected value of argument `mode`")


def unit_scale(val: float) -> Tuple[float, str]:
    """Rescale value using scale units

    Args:
        val: input value
    Returns:
        tuple of rescaled value and unit
    """

    if val // 1e12 > 0:
        return val / 1e12, 'T'
    elif val // 1e9 > 0:
        return val / 1e9, 'G'
    elif val // 1e6 > 0:
        return val / 1e6, 'M'
    elif val // 1e3 > 0:
        return val / 1e3, 'k'
    else:
        return val, ''


def format_s(f_string, min_w: Optional[int] = None, max_w: Optional[int] = None) -> str:
    if isinstance(min_w, int):
        f_string = f"{f_string:<{min_w}}"
    if isinstance(max_w, int):
        f_string = f"{f_string:.{max_w}}"

    return f_string


def format_line_str(
    layer: Dict[str, Any],
    col_w: Optional[List[int]] = None,
    wrap_mode: str = 'mid'
) -> List[str]:

    if not isinstance(col_w, list):
        col_w = [None] * 7  # type: ignore[list-item]

    max_len = col_w[0] + 3 if isinstance(col_w[0], int) else 100
    line_str = [format_s(wrap_string(format_name(layer['name'], layer['depth']), max_len, mode=wrap_mode),
                         col_w[0], col_w[0])]
    line_str.append(format_s(layer['type'], col_w[1], col_w[1]))
    line_str.append(format_s(str(layer['output_shape']), col_w[2], col_w[2]))
    line_str.append(format_s(f"{layer['grad_params'] + layer['nograd_params'] + layer['num_buffers']:,}",
                             col_w[3], col_w[3]))

    return line_str


def flops_mem(module_info):
    """Get flops and memory access summary for an expected input tensor shape

    Args:
        module_info: dictionary output of `crawl_module`
    Returns:
        (flops, mem_access)
    """
    flops = sum(layer['flops'] for layer in module_info['layers'])
    mem = sum(layer['mem'] for layer in module_info['layers'])

    return (flops, mem)

def format_info(
    module_info: Dict[str, Any],
    wrap_mode: str = 'mid'
) -> str:
    """Print module summary for an expected input tensor shape

    Args:
        module_info: dictionary output of `crawl_module`
        wrap_mode: wrapping mode
    Returns:
        formatted information
    """

    # Set margin between cols
    margin = 4
    # Dynamic col width
    # Init with headers
    headers = ['Layer', 'Type', 'Output Shape', 'Param #']
    max_w = [27, 20, 25, 15, 15, 16, 17]
    col_w = [len(s) for s in headers]
    for layer in module_info['layers']:
        col_w = [max(v, len(s))
                 for v, s in zip(col_w, format_line_str(layer, col_w=None, wrap_mode=wrap_mode))]

    # Truncate columns that are too long
    col_w = [min(v, max_v) for v, max_v in zip(col_w, max_w)]

    # Define separating lines
    line_length = sum(col_w) + (len(col_w) - 1) * margin
    thin_line = '_' * line_length
    thick_line = '=' * line_length
    dot_line = '-' * line_length

    margin_str = ' ' * margin

    # Header
    info_str = [thin_line]
    info_str.append(margin_str.join([f"{col_name:<{col_w}}" for col_name, col_w in zip(headers, col_w)]))
    info_str.append(thick_line)

    # Layers
    for layer in module_info['layers']:
        line_str = format_line_str(layer, col_w, wrap_mode)
        info_str.append((' ' * margin).join(line_str))

    # Parameter information
    info_str.append(thick_line)

    info_str.append(f"Trainable params: {module_info['overall']['grad_params']:,}")
    info_str.append(f"Non-trainable params: {module_info['overall']['nograd_params']:,}")
    num_params = module_info['overall']['grad_params'] + module_info['overall']['nograd_params']
    info_str.append(f"Total params: {num_params:,}")

    # Static RAM usage
    info_str.append(dot_line)

    # Convert to Megabytes
    param_size = (module_info['overall']['param_size'] + module_info['overall']['buffer_size']) / 1024 ** 2

    info_str.append(f"Model size (params + buffers): {param_size:.2f} Mb")

    # FLOPS information
    info_str.append(dot_line)

    flops, flops_units = unit_scale(sum(layer['flops'] for layer in module_info['layers']))
    macs, macs_units = unit_scale(sum(layer['macs'] for layer in module_info['layers']))
    mem, mem_units = unit_scale(sum(layer['mem'] for layer in module_info['layers']))

    info_str.append(f"Floating Point Operations on forward: {flops:.2f} {flops_units}FLOPs")
    info_str.append(f"Multiply-Accumulations on forward: {macs:.2f} {macs_units}MACs")
    info_str.append(f"Direct memory accesses on forward: {mem:.2f} {mem_units}mem")

    info_str.append(thin_line)

    return '\n'.join(info_str)


def aggregate_info(info: Dict[str, Any], max_depth: int) -> Dict[str, Any]:
    """Aggregate module information to a maximum depth

    Args:
        info: dictionary output of `crawl_module`
        max_depth: depth at which parent node aggregates children information
    Returns:
        edited dictionary information
    """

    if not any(layer['depth'] == max_depth for layer in info['layers']):
        raise ValueError("The `max_depth` argument cannot be higher than module depth.")

    for fw_idx, layer in enumerate(info['layers']):
        # Need to aggregate information
        if not layer['is_leaf'] and layer['depth'] == max_depth:
            grad_p, nograd_p, p_size, num_buffers, b_size = 0, 0, 0, 0, 0
            flops, macs, mem = 0, 0, 0
            for _layer in info['layers'][fw_idx + 1:]:
                # Children have superior depth and were hooked after parent
                if _layer['depth'] <= max_depth:
                    break
                # Aggregate all information (flops, macc, ram)
                flops += _layer['flops']
                macs += _layer['macs']
                mem += _layer['mem']
                grad_p += _layer['grad_params']
                nograd_p += _layer['nograd_params']
                p_size += _layer['param_size']
                num_buffers += _layer['num_buffers']
                b_size += _layer['buffer_size']
                # Take last child effective RF
                _rf, _s, _p = _layer['rf'], _layer['s'], _layer['p']

            # Update info
            info['layers'][fw_idx]['flops'] = flops
            info['layers'][fw_idx]['macs'] = macs
            info['layers'][fw_idx]['mem'] = mem
            info['layers'][fw_idx]['rf'] = _rf
            info['layers'][fw_idx]['s'] = _s
            info['layers'][fw_idx]['p'] = _p
            info['layers'][fw_idx]['grad_params'] = grad_p
            info['layers'][fw_idx]['nograd_params'] = nograd_p
            info['layers'][fw_idx]['param_size'] = p_size
            info['layers'][fw_idx]['num_buffers'] = num_buffers
            info['layers'][fw_idx]['buffer_size'] = b_size

    # Filter out further depth information
    info['layers'] = [layer for layer in info['layers'] if layer['depth'] <= max_depth]

    return info
