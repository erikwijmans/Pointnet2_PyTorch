from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import pytest
import torch
import numpy as np

pytest_plugins = ["helpers_namespace"]


def _test_loop(model, model_fn, inputs, labels):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    prev_loss = 1e10
    for _ in range(5):
        optimizer.zero_grad()
        _, loss, _ = model_fn(model, (inputs, labels))
        loss.backward()
        optimizer.step()

        assert loss.item() < prev_loss + 1.0, "Loss spiked upwards"

        prev_loss = loss.item()


@pytest.helpers.register
def cls_test_xyz(model, model_fn):
    B, N = 4, 2048
    inputs = torch.randn(B, N, 6).cuda()
    labels = torch.from_numpy(np.random.randint(0, 3, size=B)).cuda()
    model.cuda()

    _test_loop(model, model_fn, inputs, labels)


@pytest.helpers.register
def cls_test_no_xyz(model, model_fn):
    B, N = 4, 2048
    inputs = torch.randn(B, N, 3).cuda()
    labels = torch.from_numpy(np.random.randint(0, 3, size=B)).cuda()
    model.cuda()

    _test_loop(model, model_fn, inputs, labels)


@pytest.helpers.register
def semseg_test_xyz(model, model_fn):
    B, N = 4, 2048
    inputs = torch.randn(B, N, 6).cuda()
    labels = torch.from_numpy(np.random.randint(0, 3, size=B * N)).view(B, N).cuda()
    model.cuda()

    _test_loop(model, model_fn, inputs, labels)


@pytest.helpers.register
def semseg_test_no_xyz(model, model_fn):
    B, N = 4, 2048
    inputs = torch.randn(B, N, 3).cuda()
    labels = torch.from_numpy(np.random.randint(0, 3, size=B * N)).view(B, N).cuda()
    model.cuda()

    _test_loop(model, model_fn, inputs, labels)
