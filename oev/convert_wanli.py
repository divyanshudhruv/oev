"""Convert the WANLI out-of-domain test split to OEV choice-question format.

Zero-shot evaluation only: a checkpoint that never saw WANLI (td5, b77
specialists) is scored on natural-language-inference labels it must
transfer to. WANLI classes (from the dataset schema): entailment,
neutral, contradiction.

    python -m oev.convert_wanli          # writes data/wanli/test.jsonl
    python -m oev.benchmark_ext --checkpoint checkpoints_td5/oev-tiny.pt --data-dir data/wanli
"""
import json
import os

LABELS = ["entailment", "neutral", "contradiction"]

INSTRUCTIONS = "Does the hypothesis follow from the premise?"


def download_wanli():
    from datasets import load_dataset

    ds = load_dataset("alisawuffles/WANLI", split="test")
    return ds


def main(out_dir="data/wanli", limit=2000):
    os.makedirs(out_dir, exist_ok=True)
    ds = download_wanli()
    n = 0
    with open(os.path.join(out_dir, "test.jsonl"), "w", encoding="utf-8") as f:
        for row in ds:
            gold = row["gold"]
            premise = (row["premise"] or "").strip()
            hypothesis = (row["hypothesis"] or "").strip()
            state = json.dumps({"premise": premise, "hypothesis": hypothesis})
            case = {
                "id": f"wanli-{n:06d}",
                "domain": "wanli",
                "state": state,
                "questions": [{
                    "name": "relation",
                    "type": "choice",
                    "instructions": INSTRUCTIONS,
                    "options": LABELS,
                    "answer": gold,
                }],
            }
            f.write(json.dumps(case) + "\n")
            n += 1
            if n >= limit:
                break
    print(f"wrote {n} cases to {out_dir}/test.jsonl (labels: {', '.join(LABELS)})")


if __name__ == "__main__":
    main()
