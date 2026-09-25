import os

import torch


def test_smoke_train_saves_checkpoint(trained_checkpoint):
    assert os.path.exists(trained_checkpoint)
    ckpt = torch.load(trained_checkpoint, weights_only=True)
    assert "config" in ckpt and "state" in ckpt
