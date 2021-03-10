import unittest
import sys
import os
import torch.nn as nn

sys.path.append(os.path.abspath("../"))
import nnstats
import nnutils
import logging

class BasicKernelTestCase(unittest.TestCase):
    def test_conv2d(self):

        conv2d = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False).eval()
        flops, memory = nnstats.get_flops_mem(conv2d, (2, 3, 224, 224), 0)

        self.assertEqual(flops, 470450176)

    def test_rnn(self):

        rnn = nn.RNN(input_size=10, hidden_size=20).eval()
        flops, memory = nnstats.get_flops_mem(rnn, (16, 2, 10), 0)

        self.assertEqual(flops, 21120)


    # def test_auto_precision(self):

    #     class Model(torch.nn.Module):
    #         def forward(self):
    #             a = torch.randn(1)
    #             b = torch.randn(1)
    #             c = torch.cat((a, b), 0)
    #             d = torch.stack([c, c], 0)
    #             return d

    #     conv2d = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False).eval()

    #     with torch.cuda.amp.autocast(True):
    #         model()
    #         model.

if __name__ == '__main__':
        unittest.main()
