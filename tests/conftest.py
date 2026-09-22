import json
import pytest
import torch
from oev.data_gen import generate, write_splits
from oev.train import train


@pytest.fixture(scope="session")
def trained_checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("oev_artifacts")
    data_dir = root / "data"
    out_dir = root / "checkpoints"
    write_splits(generate(90, seed=21), str(data_dir))
    train(preset="tiny", epochs=1, batch_size=32, data_dir=str(data_dir), out=str(out_dir))
    return str(out_dir / "oev-tiny.pt")
