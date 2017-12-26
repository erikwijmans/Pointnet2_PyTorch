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
        loss = criterion(preds.view(labels.numel(), -1), labels.view(-1))

        _, classes = torch.max(preds.data, 2)
        acc = (classes == labels.data).sum() / labels.numel()

        return ModelReturn(preds, loss, {"acc": acc})

    return model_fn


class Pointnet2SSG(nn.Module):
    def __init__(self, num_classes, input_channels=9):
        super().__init__()

        self.initial_dropout = RandomDropout(0.4)

        self.SA_module0 = PointnetSAModule(
            npoint=1024,
            radius=0.1,
            nsample=32,
            mlp=[input_channels, 32, 32, 64])
        self.SA_module1 = PointnetSAModule(
            npoint=256, radius=0.2, nsample=32, mlp=[64 + 3, 64, 64, 128])
        self.SA_module2 = PointnetSAModule(
            npoint=64, radius=0.4, nsample=32, mlp=[128 + 3, 128, 128, 256])
        self.SA_module3 = PointnetSAModule(
            npoint=16, radius=0.8, nsample=32, mlp=[256 + 3, 256, 256, 512])

        self.FP_module0 = PointnetFPModule(mlp=[512 + 256, 256, 256])
        self.FP_module1 = PointnetFPModule(mlp=[256 + 128, 256, 256])
        self.FP_module2 = PointnetFPModule(mlp=[256 + 64, 256, 128])
        self.FP_module3 = PointnetFPModule(mlp=[128 + 6, 128, 128, 128])

        self.FC_layer = nn.Sequential(
            pt_utils.Conv1d(128, 128, bn=True), nn.Dropout(),
            pt_utils.Conv1d(128, num_classes, activation=None))

    def forward(self, xyz, points=None):
        if points is not None:
            tmp = self.initial_dropout(torch.cat([points, xyz], dim=-1))
            l0_points, l0_xyz = tmp.split(points.size(-1), dim=-1)
        else:
            l0_xyz = self.initial_dropout(xyz)
            l0_points = None

        l1_xyz, l1_points = self.SA_module0(l0_xyz, l0_points)
        l2_xyz, l2_points = self.SA_module1(l1_xyz, l1_points)
        l3_xyz, l3_points = self.SA_module2(l2_xyz, l2_points)
        l4_xyz, l4_points = self.SA_module3(l3_xyz, l3_points)

        l3_points = self.FP_module0(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.FP_module1(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.FP_module2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.FP_module3(l0_xyz, l1_xyz, l0_points,
                                    l1_points).transpose(1, 2)

        return self.FC_layer(l0_points).transpose(1, 2).contiguous()


class Pointnet2MSG(nn.Module):
    def __init__(self, num_classes, input_channels=9):
        super().__init__()

        self.initial_dropout = RandomDropout(0.95, inplace=True)
        self.initial_dropout = None

        c_in = input_channels
        self.SA_module0 = PointnetSAModuleMSG(
            npoint=1024,
            radii=[0.05, 0.1],
            nsamples=[16, 32],
            mlps=[[c_in, 16, 16, 32], [c_in, 32, 32, 64]])
        c_out_0 = 32 + 64

        c_in = c_out_0 + 3
        self.SA_module1 = PointnetSAModuleMSG(
            npoint=256,
            radii=[0.1, 0.2],
            nsamples=[16, 32],
            mlps=[[c_in, 64, 64, 128], [c_in, 64, 96, 128]])
        c_out_1 = 128 + 128

        c_in = c_out_1 + 3
        self.SA_module2 = PointnetSAModuleMSG(
            npoint=64,
            radii=[0.2, 0.4],
            nsamples=[16, 32],
            mlps=[[c_in, 128, 196, 256], [c_in, 128, 196, 256]])
        c_out_2 = 256 + 256

        c_in = c_out_2 + 3
        self.SA_module3 = PointnetSAModuleMSG(
            npoint=16,
            radii=[0.4, 0.8],
            nsamples=[16, 32],
            mlps=[[c_in, 256, 256, 512], [c_in, 256, 384, 512]])
        c_out_3 = 512 + 512

        self.FP_module3 = PointnetFPModule(mlp=[c_out_3 + c_out_2, 512, 512])
        self.FP_module2 = PointnetFPModule(mlp=[512 + c_out_1, 512, 512])
        self.FP_module1 = PointnetFPModule(mlp=[512 + c_out_0, 256, 256])
        self.FP_module0 = PointnetFPModule(
            mlp=[256 + input_channels - 3, 128, 128])

        self.FC_layer = nn.Sequential(
            pt_utils.Conv1d(128, 128, bn=True), nn.Dropout(),
            pt_utils.Conv1d(128, num_classes, activation=None))

    def forward(self, xyz, points=None):
        if points is not None and self.initial_dropout is not None:
            tmp = self.initial_dropout(torch.cat([points, xyz], dim=-1))
            points, xyz = tmp.split(points.size(-1), dim=-1)
        elif self.initial_dropout is not None:
            xyz = self.initial_dropout(xyz)

        l0_xyz, l0_points = xyz, points

        l1_xyz, l1_points = self.SA_module0(l0_xyz, l0_points)
        l2_xyz, l2_points = self.SA_module1(l1_xyz, l1_points)
        l3_xyz, l3_points = self.SA_module2(l2_xyz, l2_points)
        l4_xyz, l4_points = self.SA_module3(l3_xyz, l3_points)

        l3_points = self.FP_module3(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.FP_module2(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.FP_module1(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.FP_module0(l0_xyz, l1_xyz, l0_points,
                                    l1_points).transpose(1, 2)

        return self.FC_layer(l0_points).transpose(1, 2).contiguous()


if __name__ == "__main__":
    from torch.autograd import Variable
    import numpy as np
    import torch.optim as optim
    B = 2
    N = 32
    inputs = torch.randn(B, N, 9).cuda()
    labels = torch.from_numpy(np.random.randint(0, 3,
                                                size=B * N)).view(B, N).cuda()
    model = Pointnet2MSG(3)
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    model_fn = model_fn_decorator(nn.CrossEntropyLoss())
    for _ in range(20):
        optimizer.zero_grad()
        _, loss, _ = model_fn(model, (inputs, labels))
        loss.backward()
        print(loss.data[0])
        optimizer.step()
