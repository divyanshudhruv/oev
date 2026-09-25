import json
from typing import ClassVar

import pytest
import torch

from oev import benchmark_ext, ensemble


class _Model:
    cfg: ClassVar[dict] = {"backbone": "fake", "max_len": 8}

    def __call__(self, ids, pad_mask, anchor_pos):
        return torch.tensor([[0.0, 1.0]])


class _Packer:
    def pack(self, state, question, max_len):
        return [1, 2, 3], [1, 2], question["options"].index(question["answer"])


def _write_case(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "test.jsonl").write_text(
        json.dumps({
            "state": "state",
            "questions": [{
                "name": "choice",
                "type": "choice",
                "options": ["no", "yes"],
                "answer": "yes",
            }],
        }) + "\n",
        encoding="utf-8",
    )
    return str(data_dir)


def test_ensemble_fails_loudly_when_cuda_missing(monkeypatch, tmp_path):
    data_dir = _write_case(tmp_path)
    monkeypatch.setattr(ensemble.torch.cuda, "is_available", lambda: False)

    with pytest.raises(SystemExit, match="CUDA requested but unavailable"):
        ensemble.ensemble_accuracy(["checkpoint"], data_dir, device="cuda")


def test_ensemble_runs_when_cpu_requested_explicitly(monkeypatch, tmp_path):
    data_dir = _write_case(tmp_path)
    loaded = []

    def fake_load(checkpoint, device):
        loaded.append(device)
        return _Model()

    monkeypatch.setattr(ensemble, "load_model", fake_load)
    monkeypatch.setattr(ensemble, "HFTokenPacker", lambda name: _Packer())
    monkeypatch.setattr(ensemble.torch.cuda, "is_available", lambda: False)

    accuracy, total = ensemble.ensemble_accuracy(["checkpoint"], data_dir, device="cpu")

    assert loaded == ["cpu"]
    assert accuracy == 1.0
    assert total == 1


def test_resolve_device_fails_without_allow_cpu(monkeypatch):
    monkeypatch.setattr(benchmark_ext.torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit, match="--allow-cpu"):
        benchmark_ext.resolve_device("cuda", allow_cpu=False)


def test_resolve_device_falls_back_with_allow_cpu(monkeypatch):
    monkeypatch.setattr(benchmark_ext.torch.cuda, "is_available", lambda: False)
    assert benchmark_ext.resolve_device("cuda", allow_cpu=True) == "cpu"
    assert benchmark_ext.resolve_device("cpu", allow_cpu=False) == "cpu"


def test_manifest_records_device_and_inputs(tmp_path):
    result = {"device": "cpu", "accuracy": 0.5, "n": 2}
    path = benchmark_ext.write_manifest(
        str(tmp_path / "runs"), "ckpts/oev-tiny.pt", "data/typed", "cpu", result)
    with open(path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    assert manifest["device"] == "cpu"
    assert manifest["checkpoint"] == "ckpts/oev-tiny.pt"
    assert manifest["data_dir"] == "data/typed"
    assert manifest["metrics"]["accuracy"] == 0.5
    assert "timestamp" in manifest
