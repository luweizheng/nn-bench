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

        conv2d = nn.Conv2d(3, 8, 3)

        conv2d = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False).eval()
        flops, memory = nnstats.get_flops_mem(conv2d, (2, 3, 224, 224), 0)

        self.assertEqual(flops, 470450176)

if __name__ == '__main__':
        unittest.main()
