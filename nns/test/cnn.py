import unittest
import sys
import os
sys.path.append(os.path.abspath("../"))
import nnutils
import nnstats

import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        # input shape: 1 * 28 * 28
        self.conv = nn.Sequential(
            # conv layer 1
            # add padding: 28 * 28 -> 32 * 32
            # conv: 1 * 32 * 32 -> 6 * 28 * 28
            nn.Conv2d(in_channels=1, out_channels=6,
                      kernel_size=5, padding=2), nn.Sigmoid(),
            # 6 * 28 * 28 -> 6 * 14 * 14
            nn.MaxPool2d(kernel_size=2, stride=2),  # kernel_size, stride
            # conv layer 2
            # 6 * 14 * 14 -> 16 * 10 * 10
            nn.Conv2d(in_channels=6, out_channels=16,
                      kernel_size=5), nn.Sigmoid(),
            # 16 * 10 * 10 -> 16 * 5 * 5
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            # full connect layer 1
            nn.Linear(in_features=16 * 5 * 5, out_features=120), nn.Sigmoid(),
            # full connect layer 2
            nn.Linear(in_features=120, out_features=84), nn.Sigmoid(),
            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


class CNNTestCase(unittest.TestCase):

    def test_lenet(self):
        lenet = LeNet().eval()
        print(lenet)
        flops, memory = nnstats.get_flops_mem(lenet, (3, 1, 28, 28), 0)

        self.assertEqual(flops, 470450176)


if __name__ == '__main__':
    unittest.main()
