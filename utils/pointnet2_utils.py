import torch
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F
import torch.nn as nn
from linalg_utils import pdist2, PDist2Order
from collections import namedtuple
import _ext as pointnet2
import pytorch_utils as pt_utils
from typing import List, Tuple


class RandomDropout(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, X):
        theta = torch.Tensor(1).uniform_(0, self.p)[0]
        return pt_utils.feature_dropout_no_scaling(X, theta, self.train,
                                                   self.inplace)


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        r"""
        Uses iterative furthest point sampling to select a set of npoint points that have the largest
        minimum distance

        Parameters
        ---------
        xyz : torch.Tensor
            (B, N, 3) tensor where N > npoint
        npoint : int32
            number of points in the sampled set

        Returns
        torch.Tensor
            (B, npoint) tensor containing the set
        ------
        """
        B, N, _ = xyz.size()

        output = torch.cuda.IntTensor(B, npoint)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

        xyz = xyz.contiguous()
        temp = temp.contiguous()
        output = output.contiguous()

        pointnet2.furthest_point_sampling_wrapper(B, N, npoint, xyz, temp,
                                                  output)

        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply


class GatherPoints(Function):
    @staticmethod
    def forward(ctx, points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        r"""
        Uses iterative furthest point sampling to select a set of npoint points that have the largest
        minimum distance

        Parameters
        ---------
        points : torch.Tensor
            (B, N, 3) tensor

        idx : torch.Tensor
            (B, npoint) tensor of the points to gather

        Returns
        torch.Tensor
            (B, npoint, 3) tensor
        ------
        """

        B, N, C = points.size()
        npoint = idx.size(1)

        output = torch.cuda.FloatTensor(B, npoint, C)

        points = points.contiguous()
        idx = idx.contiguous()
        output = output.contiguous()

        pointnet2.gather_points_wrapper(B, N, C, npoint, points, idx, output)

        return output

    @staticmethod
    def backward(ctx, a=None):
        return None, None


gather_points = GatherPoints.apply


class ThreeNN(Function):
    @staticmethod
    def forward(ctx, unknown: torch.Tensor,
                known: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
            Find the three nearest neighbors of unknown in known
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of known points
        known : torch.Tensor
            (B, m, 3) tensor of unknown points

        Returns
        -------
        dist : torch.Tensor
            (B, n, 3) l2 distance to the three nearest neighbors
        idx : torch.Tensor
            (B, n, 3) index of 3 nearest neighbors
        """
        B, N, _ = unknown.size()
        m = known.size(1)
        dist2 = torch.cuda.FloatTensor(B, N, 3)
        idx = torch.cuda.IntTensor(B, N, 3)

        unknown = unknown.contiguous()
        known = known.contiguous()
        dist2 = dist2.contiguous()
        idx = idx.contiguous()
        pointnet2.three_nn_wrapper(B, N, m, unknown, known, dist2, idx)

        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply


class ThreeInterpolate(Function):
    @staticmethod
    def forward(ctx, points: torch.Tensor, idx: torch.Tensor,
                weight: torch.Tensor) -> torch.Tensor:
        r"""
            Performs weight linear interpolation on 3 points
        Parameters
        ----------
        points : torch.Tensor
            (B, m, c)  Points to be interpolated from
        idx : torch.Tensor
            (B, n, 3) three nearest neighbors of the target points in points
        weight : torch.Tensor
            (B, n, 3) weights

        Returns
        -------
        torch.Tensor
            (B, n, c) tensor of the interpolated points
        """

        B, m, c = points.size()
        n = idx.size(1)

        ctx.three_interpolate_for_backward = (idx, weight, m)

        output = torch.cuda.FloatTensor(B, n, c)

        points = points.contiguous()
        idx = idx.contiguous()
        weight = weight.contiguous()
        output = output.contiguous()
        pointnet2.three_interpolate_wrapper(B, m, c, n, points, idx, weight,
                                            output)

        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Parameters
        ----------
        grad_out : torch.Tensor
            (B, n, c) tensor with gradients of ouputs

        Returns
        -------
        grad_points : torch.Tensor
            (B, m, c) tensor with gradients of points
        None

        None
        """
        idx, weight, m = ctx.three_interpolate_for_backward
        B, n, c = grad_out.size()

        grad_points = Variable(torch.cuda.FloatTensor(B, m, c).zero_())

        grad_out = grad_out.contiguous()
        idx = idx.contiguous()
        weight = weight.contiguous()
        grad_points = grad_points.contiguous()
        pointnet2.three_interpolate_grad_wrapper(B, n, c, m, grad_out.data,
                                                 idx, weight, grad_points.data)

        return grad_points, None, None


three_interpolate = ThreeInterpolate.apply


class GroupPoints(Function):
    @staticmethod
    def forward(ctx, points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        r"""

        Parameters
        ----------
        points : torch.Tensor
            (B, N, C) tensor of points to group
        idx : torch.Tensor
            (B, npoint, nsample) tensor containing the indicies of points to group with

        Returns
        -------
        torch.Tensor
            (B, npoint, nsample, C) tensor
        """
        B, npoints, nsample = idx.size()
        _, N, C = points.size()

        output = torch.cuda.FloatTensor(B, npoints, nsample, C)

        points = points.contiguous()
        idx = idx.contiguous()
        output = output.contiguous()
        pointnet2.group_points_wrapper(B, N, C, npoints, nsample, points, idx,
                                       output)

        ctx.idx_N_C_for_backward = (idx, N, C)
        return output

    @staticmethod
    def backward(ctx,
                 grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""

        Parameters
        ----------
        grad_out : torch.Tensor
            (B, npoint, nsample, C) tensor of the gradients of the output from forward

        Returns
        -------
        torch.Tensor
            (B, N, C) gradient of the points
        None
        """
        idx, N, C = ctx.idx_N_C_for_backward

        B, npoint, nsample, _ = grad_out.size()
        grad_points = Variable(torch.cuda.FloatTensor(B, N, C).zero_())

        grad_out = grad_out.contiguous()
        grad_points = grad_points.contiguous()
        pointnet2.group_points_grad_wrapper(
            B, N, C, npoint, nsample, grad_out.data, idx, grad_points.data)

        return grad_points, None


group_points = GroupPoints.apply


class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor,
                new_xyz: torch.Tensor) -> torch.Tensor:
        r"""

        Parameters
        ---------
        radius : float
            radius of the balls
        nsample : int
            maximum number of points in the balls
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the points
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the ball query

        Returns
        ------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the points that form the query balls
        """

        B, N, _ = xyz.size()
        npoint = new_xyz.size(1)
        idx = torch.cuda.IntTensor(B, npoint, nsample).zero_()

        new_xyz = new_xyz.contiguous()
        xyz = xyz.contiguous()
        idx = idx.contiguous()
        pointnet2.ball_query_wrapper(B, N, npoint, radius, nsample, new_xyz,
                                     xyz, idx)

        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply


class QueryAndGroup(nn.Module):
    r"""
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of points to gather in the ball
    """

    def __init__(self, radius: float, nsample: int, use_xyz: bool = True):
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(
            self,
            xyz: torch.Tensor,
            new_xyz: torch.Tensor,
            points: torch.Tensor = None) -> Tuple[torch.Tensor]:
        r"""
        Parameters
        ---------
        xyz : torch.Tensor
            xyz coordinates of the points (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        points : torch.Tensor
            Descriptors of the points (B, N, C)

        Returns
        -------
        new_points : torch.Tensor
            (B, npoint, nsample, 3 + C) tensor
        """

        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = group_points(xyz, idx)  # (B, npoint, nsample, 3)
        grouped_xyz -= new_xyz.unsqueeze(2)

        if points is not None:
            grouped_points = group_points(points, idx)
            if self.use_xyz:
                new_points = torch.cat(
                    [grouped_xyz, grouped_points],
                    dim=-1)  # (B, npoint, nsample, 3 + C)
            else:
                new_points = group_points
        else:
            new_points = grouped_xyz

        return new_points


class GroupAll(nn.Module):
    r"""
    Groups all points

    Parameters
    ---------
    """

    def __init__(self, use_xyz: bool = True):
        super().__init__()
        self.use_xyz = use_xyz

    def forward(
            self,
            xyz: torch.Tensor,
            new_xyz: torch.Tensor,
            points: torch.Tensor = None) -> Tuple[torch.Tensor]:
        r"""
        Parameters
        ---------
        xyz : torch.Tensor
            xyz coordinates of the points (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        points : torch.Tensor
            Descriptors of the points (B, N, C)

        Returns
        -------
        new_points : torch.Tensor
            (B, npoint, nsample, 3 + C) tensor
        """

        grouped_xyz = xyz.view(xyz.size(0), 1, xyz.size(1), xyz.size(2))
        if points is not None:
            grouped_points = points.view(points.size(0), 1, points.size(1), points.size(2))
            if self.use_xyz:
                new_points = torch.cat(
                    [grouped_xyz, grouped_points],
                    dim=-1)  # (B, npoint, nsample, 3 + C)
            else:
                new_points = group_points
        else:
            new_points = grouped_xyz

        return new_points
