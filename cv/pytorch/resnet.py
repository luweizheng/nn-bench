import torch
import torch.optim as optim
import torch.nn as nn
import time
import argparse
import ssl
import torchvision
import sys
import os

sys.path.append(os.path.abspath("../../nns"))
import nnutils
import nnstats

ssl._create_default_https_context = ssl._create_unverified_context

def build_model():
    model = torchvision.models.resnet50(pretrained=True)
    return model


def get_raw_data():
    input_tensor = torch.randn(2, 3, 224, 224)
    return input_tensor


def criterion(x, y=None):
    return x.sum()


def main(args):
    # 1.准备工作
    if args.platform == "gpu":
        device = torch.device('cuda:0')
        device_func = torch.cuda
        prof_kwargs = {'use_cuda': True}
    elif args.platform == "npu":
        device = torch.device('npu:0')
        device_func = torch.npu
        prof_kwargs = {'use_npu': True}
    else:
        device = torch.device('cpu')
    print("Running on device {}".format(device))
    # if args.device.startswith('cuda'):
    #     torch.cuda.set_device(args.device)
    #     prof_kwargs = {'use_cuda': True}
    # elif args.device.startswith('npu'):
    #     torch.npu.set_device(args.device)
    #     prof_kwargs = {'use_npu': True}
    # else:
    #     prof_kwargs = {}


    input_tensor = get_raw_data()
    input_tensor = input_tensor.to(device)

    # 2.构建模型
    model = build_model()
    model.eval()
    flops, mem = nnstats.get_flops_mem(model, (2, 3, 224, 224))

    if args.FusedSGD:
        from apex.optimizers import NpuFusedSGD
        optimizer = NpuFusedSGD(model.parameters(), lr=0.01)
        model = model.to(device)
        model = model.half()
        if args.amp:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level,
                                              loss_scale=None if args.loss_scale == -1 else args.loss_scale,
                                              combine_grad=True)
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        model = model.to(device)
        if args.amp:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level,
                                              loss_scale=None if args.loss_scale == -1 else args.loss_scale)


    # 先运行一次，保证prof得到的性能是正确的
    def run():
        output_tensor = model(input_tensor)
        loss = criterion(output_tensor)
        optimizer.zero_grad()
        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        return loss
    
    # warm up
    for i in range(10):
        loss = run()

    # bench
    start_event = device_func.Event(enable_timing=True)
    end_event = device_func.Event(enable_timing=True)
    start_event.record()
    for i in range(100):
        loss = run()
    end_event.record()
    device_func.synchronize()
    elapsed_time = start_event.elapsed_time(end_event) / 1000
    flop_sec = flops * 3 * 100 / elapsed_time
    flop_sec_scaled, flop_sec_unit = nnutils.unit_scale(flop_sec)
    print(f"flops: {flop_sec}")
    print(f"time: {elapsed_time:.3f}")
    print(f"flops_scaled: {flop_sec_scaled} {flop_sec_unit}")


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

    args = parser.parse_args()

    main(args)