"""Convert the DAIR Emotion test split to OEV choice-question format.

Zero-shot evaluation only: this writes the *test* set so a checkpoint that
never saw emotion data (e.g. the typed-decisions specialist) can be scored
without any fine-tuning. Run:

    python -m oev.convert_emotion            # writes data/emotion/test.jsonl
    python -m oev.benchmark_ext --checkpoint checkpoints_td5/oev-tiny.pt --data-dir data/emotion
"""
import json
import os

LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

INSTRUCTIONS = "Which emotion does this text express?"


def download_emotion():
    from datasets import load_dataset

    ds = load_dataset("dair-ai/emotion", split="test")
    assert ds.features["label"].names == LABELS, "label order drifted from the published schema"
    return ds


def main(out_dir="data/emotion"):
    os.makedirs(out_dir, exist_ok=True)
    ds = download_emotion()
    n = 0
    with open(os.path.join(out_dir, "test.jsonl"), "w", encoding="utf-8") as f:
        for row in ds:
            case = {
                "id": f"emotion-{n:06d}",
                "domain": "emotion",
                "state": row["text"],
                "questions": [{
                    "name": "emotion",
                    "type": "choice",
                    "instructions": INSTRUCTIONS,
                    "options": LABELS,
                    "answer": LABELS[row["label"]],
                }],
            }
            f.write(json.dumps(case) + "\n")
            n += 1
    print(f"wrote {n} cases to {out_dir}/test.jsonl (labels: {', '.join(LABELS)})")


if __name__ == "__main__":
    main()
