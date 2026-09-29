"""Convert the DAIR Emotion test split for zero-shot eval only.

    python -m oev.convert_emotion
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
