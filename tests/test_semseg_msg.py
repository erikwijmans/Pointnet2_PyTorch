from pointnet2.models.pointnet2_msg_sem import model_fn_decorator, Pointnet2MSG
import torch.nn as nn
import pytest


def test_xyz():
    model = Pointnet2MSG(3, input_channels=3)
    pytest.helpers.semseg_test_xyz(model, model_fn_decorator(nn.CrossEntropyLoss()))


def test_no_xyz():
    model = Pointnet2MSG(3, input_channels=0)
    pytest.helpers.semseg_test_no_xyz(model, model_fn_decorator(nn.CrossEntropyLoss()))
