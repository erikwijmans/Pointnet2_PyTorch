import torch
import torch.nn as nn
import torch.nn.functional as F

import pointnet2_utils
import pytorch_utils as pt_utils
from typing import List


class _PointnetSAModuleBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None

    def forward(self, xyz: torch.Tensor,
                points: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the points
        points : torch.Tensor
            (B, N, C) tensor of the descriptors of the the points

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new points' xyz
        new_points : torch.Tensor
            (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_points descriptors
        """

        new_points_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        new_xyz = pointnet2_utils.gather_points(
            xyz_flipped,
            pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        ).transpose(1, 2).contiguous() if self.npoint is not None else None

        for i in range(len(self.groupers)):
            new_points = self.groupers[i](
                xyz, new_xyz, points
            )  # (B, C, npoint, nsample)

            new_points = self.mlps[i](
                new_points
            )  # (B, mlp[-1], npoint, nsample)
            new_points = F.max_pool2d(
                new_points, kernel_size=[1, new_points.size(3)]
            )  # (B, mlp[-1], npoint, 1)
            new_points = new_points.squeeze(-1)  # (B, mlp[-1], npoint)

            new_points_list.append(new_points)

        return new_xyz, torch.cat(new_points_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    r"""Pointnet set abstrction layer with multiscale grouping

    Parameters
    ----------
    npoint : int
        Number of points
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """

    def __init__(
            self,
            *,
            npoint: int,
            radii: List[float],
            nsamples: List[int],
            mlps: List[List[int]],
            bn: bool = True,
            use_xyz: bool = True
    ):
        super().__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=bn))


class PointnetSAModule(_PointnetSAModuleBase):
    r"""Pointnet set abstrction layer

    Parameters
    ----------
    npoint : int
        Number of points
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet before the global max_pool
    bn : bool
        Use batchnorm
    """

    def __init__(
            self,
            *,
            mlp: List[int],
            npoint: int = None,
            radius: float = None,
            nsample: int = None,
            bn: bool = True,
            use_xyz: bool = True
    ):
        super().__init__()
        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()

        if self.npoint is not None:
            assert radius is not None
            assert nsample is not None
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
            )
        else:
            self.groupers.append(pointnet2_utils.GroupAll(use_xyz=use_xyz))

        if use_xyz:
            mlp[0] += 3

        self.mlps.append(pt_utils.SharedMLP(mlp, bn=bn))


class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another

    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """

    def __init__(self, *, mlp: List[int], bn: bool = True):
        super().__init__()
        self.mlp = pt_utils.SharedMLP(mlp, bn=bn)

    def forward(
            self, unknown: torch.Tensor, known: torch.Tensor,
            unknow_feats: torch.Tensor, known_feats: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown points
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known points
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propigated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propigated

        Returns
        -------
        new_points : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown points
        """

        dist, idx = pointnet2_utils.three_nn(unknown, known)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm

        interpolated_feats = pointnet2_utils.three_interpolate(
            known_feats, idx, weight
        )
        if unknow_feats is not None:
            new_points = torch.cat([interpolated_feats, unknow_feats],
                                   dim=1)  #(B, C2 + C1, n)
        else:
            new_points = interpolated_feats

        new_points = new_points.unsqueeze(-1)
        new_points = self.mlp(new_points)

        return new_points.squeeze(-1)


if __name__ == "__main__":
    from torch.autograd import Variable
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    xyz = Variable(torch.randn(2, 9, 3).cuda(), requires_grad=True)
    xyz_feats = Variable(torch.randn(2, 9, 6).cuda(), requires_grad=True)

    test_module = PointnetSAModuleMSG(
        npoint=2, radii=[5.0, 10.0], nsamples=[6, 3], mlps=[[9, 3], [9, 6]]
    )
    test_module.cuda()
    print(test_module(xyz, xyz_feats))

    #  test_module = PointnetFPModule(mlp=[6, 6])
    #  test_module.cuda()
    #  from torch.autograd import gradcheck
    #  inputs = (xyz, xyz, None, xyz_feats)
    #  test = gradcheck(test_module, inputs, eps=1e-6, atol=1e-4)
    #  print(test)

    for _ in range(1):
        _, new_points = test_module(xyz, xyz_feats)
        new_points.backward(torch.cuda.FloatTensor(*new_points.size()).fill_(1))
        print(new_points)
        print(xyz.grad)
