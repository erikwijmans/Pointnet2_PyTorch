import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../utils"))

import torch
import torch.nn as nn
from torch.autograd import Variable
import pytorch_utils as pt_utils
from pointnet2_modules import PointnetSAModule, PointnetFPModule, PointnetSAModuleMSG
from pointnet2_utils import RandomDropout
from collections import namedtuple


def model_fn_decorator(criterion):
    ModelReturn = namedtuple("ModelReturn", ['preds', 'loss', 'acc'])

    def model_fn(model, data, epoch=0, eval=False):
        inputs, labels = data
        inputs = Variable(inputs.cuda(async=True), volatile=eval)
        labels = Variable(labels.cuda(async=True), volatile=eval)

        xyz = inputs[..., :3]
        if inputs.size(2) > 3:
            points = inputs[..., 3:]
        else:
            points = None

        preds = model(xyz, points)
        labels = labels.view(-1)
        loss = criterion(preds, labels)

        _, classes = torch.max(preds.data, -1)
        acc = (classes == labels.data).sum() / labels.numel()

        return ModelReturn(preds, loss, {"acc": acc})

    return model_fn


class Pointnet2SSG(nn.Module):

    def __init__(self, num_classes, input_channels=3, use_xyz=True):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.2,
                nsample=64,
                mlp=[input_channels, 64, 64, 128],
                use_xyz=use_xyz
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128, radius=0.4, nsample=64, mlp=[128, 128, 128, 256]
            )
        )
        self.SA_modules.append(PointnetSAModule(mlp=[256, 256, 512, 1024]))

        self.FC_layer = nn.Sequential(
            pt_utils.FC(1024, 512, bn=True),
            nn.Dropout(p=0.5),
            pt_utils.FC(512, 256, bn=True),
            nn.Dropout(p=0.5),
            pt_utils.FC(256, num_classes, activation=None)
        )

    def forward(self, xyz, points=None):
        xyz = xyz.contiguous()
        points = (
            points.transpose(1, 2).contiguous() if points is not None else None
        )
        for module in self.SA_modules:
            xyz, points = module(xyz, points)

        return self.FC_layer(points.squeeze(-1))


if __name__ == "__main__":
    from torch.autograd import Variable
    import numpy as np
    import torch.optim as optim
    import torch.autograd.profiler as profiler
    B = 2
    N = 2048
    inputs = torch.randn(B, N, 6).cuda()
    labels = torch.from_numpy(np.random.randint(0, 3, size=B)).cuda()
    model = Pointnet2SSG(3, input_channels=3)
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    model_fn = model_fn_decorator(nn.CrossEntropyLoss())
    for _ in range(20):
        optimizer.zero_grad()
        _, loss, _ = model_fn(model, (inputs, labels))
        loss.backward()
        print(loss.data[0])
        optimizer.step()

    # use_xyz=False
    inputs = torch.randn(B, N, 3).cuda()
    labels = torch.from_numpy(np.random.randint(0, 3, size=B)).cuda()
    model = Pointnet2SSG(3, input_channels=3, use_xyz=False)
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    model_fn = model_fn_decorator(nn.CrossEntropyLoss())
    for _ in range(20):
        optimizer.zero_grad()
        _, loss, _ = model_fn(model, (inputs, labels))
        loss.backward()
        print(loss.data[0])
        optimizer.step()
