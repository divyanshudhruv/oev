import json
import pytest
import torch
from oev.evaluate import evaluate


@pytest.fixture(scope="module")
def mini_checkpoint(tmp_path_factory, trained_checkpoint):
    return trained_checkpoint


def test_evaluate_runs_end_to_end_cpu(trained_checkpoint, tmp_path):
    root = tmp_path / "evaldata"
    root.mkdir()
    rows = [
        {"id": f"review-{i:04d}", "domain": "review", "state": f"it was {'great' if i % 2 else 'terrible'}", "questions": [{"name": "sentiment", "type": "choice", "options": ["positive", "negative", "neutral"], "answer": "positive" if i % 2 else "negative"}]}
        for i in range(12)
    ]
    for split, n in (("train", 8), ("valid", 2), ("test", 2)):
        with open(root / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows[:n]:
                f.write(json.dumps(r) + "\n")
    results = evaluate(str(trained_checkpoint), data_dir=str(root), batch_size=4)
    assert 0.0 <= results["accuracy_choice"] <= 1.0
    assert results["latency_ms_1q"] > 0
    assert results["latency_ms_per_q_batched"] > 0
