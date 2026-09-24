"""Convert PolyAI/banking77 into OEV format.

Banking77: 13,083 customer-service queries, 77 fine-grained intents.
This is the high-cardinality test: 77 options in ONE choice question.
Laya's published number is 0.425 (head token budget squeezes each label
to 3-4 tokens); Jev publishes 0.870. Our anchor mechanism gives every
option full tokens, so this is our most winnable remaining benchmark row.

One choice question per state, all 77 intents as options with their
human-readable intent names as option text.

Usage:
    python -m oev.convert_banking77
Writes:
    data/banking77/train.jsonl  (~10k cases, 1 question each)
    data/banking77/valid.jsonl  (1,000 cases)
    data/banking77/test.jsonl   (3,000 cases)
"""

import json
from pathlib import Path


def _read_csv(url, names):
    """Download one PolyAI banking csv (columns: text,category) into row dicts."""
    import csv
    import io
    import urllib.request

    raw = urllib.request.urlopen(url, timeout=60).read().decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(raw)))
    for r in rows:
        r["label"] = names.index(r["category"])
    return rows


def download_banking77(names):
    # the HF repo ships a legacy loading script that modern `datasets`
    # refuses to run, and the auto-converted parquet branch was removed;
    # the canonical source is PolyAI's own github csvs
    base = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"
    return _read_csv(f"{base}/train.csv", names), _read_csv(f"{base}/test.csv", names)


def intent_names():
    """Official BANKING77 intent names in the HF dataset's label-id order,
    fetched from the dataset's schema. Checkpoints are trained against this
    exact id order, so it must never be replaced by another source."""
    import json
    import urllib.request

    url = "https://huggingface.co/datasets/PolyAI/banking77/resolve/main/dataset_infos.json"
    data = json.load(urllib.request.urlopen(url, timeout=30))
    names = list(data.values())[0]["features"]["label"]["names"]
    assert len(names) == 77, f"expected 77 official intents, got {len(names)}"
    return names


def write_jsonl(rows, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def convert(rows, names):
    """One case per query: a single choice question with ALL 77 intents."""
    n = len(names)
    for r in rows:
        label = int(r["label"])
        if not (0 <= label < n):
            continue
        yield {
            "id": f"b77-{r.get('id', label)}",
            "domain": "banking77",
            "state": r["text"],
            "questions": [{
                "name": "intent",
                "type": "choice",
                "instructions": "Which banking intent does this customer query belong to? Choose the single closest match.",
                "options": names,
                "answer": names[label],
            }],
        }


if __name__ == "__main__":
    names = intent_names()
    assert len(names) == 77, f"expected 77 intents, got {len(names)}"
    train, test = download_banking77(names)

    train_rows = list(convert(train, names))
    # carve a validation split off the train pool
    write_jsonl(train_rows[:-1000], "data/banking77/train.jsonl")
    write_jsonl(train_rows[-1000:], "data/banking77/valid.jsonl")
    test_rows = list(convert(test, names))
    write_jsonl(test_rows, "data/banking77/test.jsonl")
    print(f"banking77 train/valid/test written: {len(train_rows) - 1000}/{1000}/{len(test_rows)} cases (77 options each)")
