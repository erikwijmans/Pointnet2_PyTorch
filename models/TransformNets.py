import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import pytorch_utils as pt_utils


class TransformNet(nn.Module):
    def __init__(self, in_size, channels, K, scale=False):
        super().__init__()
        self.K, self.scale = K, scale

        self.convs = nn.Sequential()
        self.convs.add_module('conv0',
                              pt_utils.Conv2d(
                                  in_size, 64, kernel_size=[1, channels], bn=True))
        self.convs.add_module('rest',
                              pt_utils.SharedMLP([64, 128, 1024], bn=True))

        self.fc = nn.Sequential(
            pt_utils.FC(1024, 512, bn=True), pt_utils.FC(512, 256, bn=True))

        outsize = K * K
        if scale:
            outsize += 1

        self.final_W = nn.Parameter(torch.FloatTensor(256, outsize))
        self.final_b = nn.Parameter(torch.FloatTensor(outsize))

        self.init_weights()

    def forward(self, X):
        X = self.convs(X)
        X = F.adaptive_max_pool2d(X, [1, 1])
        X = self.fc(X.view(-1, 1024))
        X = X @ self.final_W + self.final_b

        rotation = X[:, 0:self.K * self.K].contiguous().view(
            -1, self.K, self.K)

        if not self.scale:
            return rotation, None

        scale = X[:, -1].contiguous()

        return rotation, scale

    def init_weights(self):
        torch.nn.init.constant(self.final_W, 0)
        self.final_b.data[:self.K * self.K] = (torch.eye(
            self.K, self.K) + 1e-1 * torch.randn(self.K, self.K)).view(-1)
        if self.scale:
            self.final_b.data[-1] = 1.0


class TranslationNet(nn.Module):
    def forward(self, X):
        return -torch.mean(X, dim=1)


if __name__ == "__main__":
    from torch.autograd import Variable
    net = TransformNet(5, 1, 3, True)
    net.init_weights()
    data = Variable(torch.FloatTensor(1, 5, 10, 1))
    print(net(data))

    net = TranslationNet(5, 1, 3)
    net.init_weights()
    print(net(data))
