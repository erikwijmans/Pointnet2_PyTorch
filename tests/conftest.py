import os

import hydra
import hydra.experimental
import numpy as np
import pytest
import torch

pytest_plugins = ["helpers_namespace"]

hydra.experimental.initialize(
    os.path.join(os.path.dirname(__file__), "../pointnet2/config")
)


@pytest.helpers.register
def build_cfg(overrides=[]):
    return hydra.experimental.compose("config.yaml", overrides)


@pytest.helpers.register
def get_model(overrides=[]):
    cfg = build_cfg(overrides)
    return hydra.utils.instantiate(cfg.task_model, cfg)


def _test_loop(model, inputs, labels):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    prev_loss = 1e10
    for _ in range(5):
        optimizer.zero_grad()
        res = model.training_step((inputs, labels), None)
        loss = res["loss"]
        loss.backward()
        optimizer.step()

        assert loss.item() < prev_loss + 1.0, "Loss spiked upwards"

        prev_loss = loss.item()


@pytest.helpers.register
def cls_test(model):
    B, N = 4, 2048
    inputs = torch.randn(B, N, 6).cuda()
    labels = torch.from_numpy(np.random.randint(0, 3, size=B)).cuda()
    model.cuda()

    _test_loop(model, inputs, labels)


@pytest.helpers.register
def semseg_test(model):
    B, N = 4, 2048
    inputs = torch.randn(B, N, 9).cuda()
    labels = torch.from_numpy(np.random.randint(0, 3, size=B * N)).view(B, N).cuda()
    model.cuda()

    _test_loop(model, inputs, labels)
