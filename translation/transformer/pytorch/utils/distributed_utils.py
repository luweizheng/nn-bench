import pickle

import torch.distributed

from utils import utils

def is_master(args):
    return args.distributed_rank == 0


def distributed_init(args):
    if args.distributed_world_size == 1:
        raise ValueError('Cannot initialize distributed with distributed_world_size=1')

    print('| distributed init (rank {}): {}'.format(
        args.distributed_rank, args.distributed_init_method), flush=True)
    if args.distributed_init_method.startswith('tcp://'):
        torch.distributed.init_process_group(
            backend=args.distributed_backend, init_method=args.distributed_init_method,
            world_size=args.distributed_world_size, rank=args.distributed_rank)
    elif args.distributed_init_method.startswith('env://'):
        import os
        print("| distributed env init. MASTER_ADDR: " + os.environ['MASTER_ADDR'] + ", MASTER_PORT: " + os.environ['MASTER_PORT'] +
                ", WORLD_SIZE: " + os.environ['WORLD_SIZE'] + ", RANK: " + os.environ['RANK'], flush=True)
        torch.distributed.init_process_group(
            backend=args.distributed_backend, init_method=args.distributed_init_method)
        print("| distributed init done!", flush=True)
        args.distributed_world_size = int(os.environ['WORLD_SIZE'])
    else:
        torch.distributed.init_process_group(
            backend=args.distributed_backend, init_method=args.distributed_init_method,
            world_size=args.distributed_world_size)

    args.distributed_rank = torch.distributed.get_rank()
    suppress_output(args)

    return args.distributed_rank


def suppress_output(main_args):
    """Suppress printing on the current device. Force printing with `force=True`."""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print_master(*args, **kwargs):
        if 'force' in kwargs:
            force = kwargs.pop('force')
        builtin_print(*args, **kwargs)


    def print(*args, **kwargs):
        if 'force' in kwargs:
            force = kwargs.pop('force')
            if force:
                builtin_print(*args, **kwargs)
    if(is_master(main_args)):
        __builtin__.print = print_master
    else:
        __builtin__.print = print

def all_gather_list(data, max_size=16384):
    """Gathers arbitrary data from all nodes into a list."""
    world_size = torch.distributed.get_world_size()
    if not hasattr(all_gather_list, '_in_buffer') or \
            max_size != len(all_gather_list._in_buffer):
        all_gather_list._in_buffer = torch.CharTensor(max_size).npu()
        all_gather_list._out_buffers = [
            torch.CharTensor(max_size).npu()
            for i in range(world_size)
        ]
    in_buffer = all_gather_list._in_buffer
    out_buffers = all_gather_list._out_buffers

    enc = pickle.dumps(data)
    enc_size = len(enc)
    if enc_size + 2 > max_size:
        raise ValueError('encoded data exceeds max_size: {}'.format(enc_size + 2))
    assert max_size < 255*256
    in_buffer[0] = enc_size // 255 - 128  # this encoding works for max_size < 65k
    in_buffer[1] = enc_size % 255 - 128
    # Hccl only support signed int8; need to add 128 to be unsighed int8
    in_buffer[2:enc_size+2] = torch.CharTensor(list(map(lambda x : x - 128, list(enc)))).npu()

    torch.distributed.all_gather(out_buffers, in_buffer)
    temp = [
            torch.ByteTensor(max_size).npu()
            for i in range(world_size)
           ]
    for i in range(len(out_buffers)):
        temp[i] = out_buffers[i] + 128
    result = []
    for i in range(world_size):
        t = temp[i]
        size = (255 * utils.item(t[0])) + utils.item(t[1])
        result.append(
            pickle.loads(bytes(t[2:size+2].tolist()))
        )

    # result = []
    # for i in range(world_size):
    #     out_buffer = out_buffers[i]
    #     size = (255 * utils.item(out_buffer[0])) + utils.item(out_buffer[1])
    #     result.append(
    #         pickle.loads(bytes(out_buffer[2:size+2].tolist()))
    #     )
    return result
