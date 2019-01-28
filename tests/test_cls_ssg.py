from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
from pointnet2.models.pointnet2_ssg_cls import model_fn_decorator, Pointnet2SSG
import torch.nn as nn
import pytest


def test_xyz():
    model = Pointnet2SSG(3, input_channels=3)
    pytest.helpers.cls_test_xyz(model, model_fn_decorator(nn.CrossEntropyLoss()))


def test_no_xyz():
    model = Pointnet2SSG(3, input_channels=0)
    pytest.helpers.cls_test_no_xyz(model, model_fn_decorator(nn.CrossEntropyLoss()))
