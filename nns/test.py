import torch.nn as nn
import nnstats
from torchvision.models import resnet18, densenet121
import nnutils
import logging

# linear = nn.Linear(in_features=64, out_features=128).eval()
# module_info = nns.crawl_module(linear, (1, 64))
# print(module_info)
conv2d = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False).eval()
flops, memory = nnstats.get_flops_mem(conv2d, (2, 3, 224, 224), 0)
logging.debug(nnutils.unit_scale(flops))
logging.debug(nnutils.unit_scale(memory))

# rnn = nn.RNN(input_size=100, hidden_size=256).eval()
# flops, memory = nnstats.get_flops_mem(rnn, (4, 1, 100), 0)
# print(flops)
# print(nnutils.unit_scale(flops))

# stats = crawler.crawl_module(conv2d, (128, 3, 224, 224))
# module_info = nnutils.aggregate_info(stats, 0)
# print(nnutils.flops_dmas(module_info))

# resnet18 = resnet18().eval()
# print(nnstats.get_flops_dmas(resnet18, (1, 3, 224, 224)))

# densenet121 = densenet121().eval()
# print(nnstats.get_flops_dmas(densenet121, (1, 3, 224, 224)))