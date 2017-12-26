import torch
import numpy as np


class PointcloudScale(object):
    def __init__(self, mean=2.0, std=1.0, clip=1.8):
        self.mean, self.std, self.clip = mean, std, clip

    def __call__(self, points):
        scaler = points.new(1).normal_(
            mean=self.mean, std=self.std).clamp_(
                max(self.mean - self.clip, 0.01), self.mean + self.clip)
        return scaler * points


class PointcloudRotate(object):
    def __init__(self, x_axis=False, z_axis=True):
        assert x_axis or z_axis
        self.x, self.y = x_axis, z_axis

    def _get_angles(self):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)

        return cosval, sinval

    def __call__(self, points):
        if self.z:
            sinval, cosval = self._get_angles()
            Rz = points.new([[cosval, sinval, 0], [-sinval, cosval, 0],
                             [0, 0, 1]])
        else:
            Rz = torch.eye(3)

        if self.x:
            sinval, cosval = self._get_angles()
            Rx = points.new([[1, 0, 0], [0, cosval, sinval],
                             [0, -sinval, cosval]])
        else:
            Rx = torch.eye(3)

        rot_mat = Rx @ Rz

        return points @ rot_mat


class PointcloudJitter(object):
    def __init__(self, std=0.01, clip=0.03):
        self.std, self.clip = std, clip

    def __call__(self, points):
        jittered_data = points.new(*points.size()).normal_(
            mean=0.0, std=self.std).clamp_(-self.clip, self.clip)
        return points + jittered_data


class PointcloudTranslate(object):
    def __init__(self, std=1.0, clip=3.0):
        self.std, self.clip = std, clip

    def __call__(self, points):
        translation = points.new(3).normal_(
            mean=0.0, std=self.std).clamp_(-self.clip, self.clip)
        return points + translation


class PointcloudToTensor(object):
    def __call__(self, points):
        return torch.from_numpy(points).float()
