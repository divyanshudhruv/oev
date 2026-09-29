"""Convert PolyAI/banking77 into OEV format: all 77 intents as anchors.

    python -m oev.convert_banking77
Writes data/banking77/{train,valid,test}.jsonl.
"""

import json
from pathlib import Path


def _read_csv(url, names):
    # download one PolyAI banking csv (columns: text,category)
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
    # official intent names in HF label-id order; checkpoints train against
    # this exact order, so never replace the source
    import json
    import urllib.request

    url = "https://huggingface.co/datasets/PolyAI/banking77/resolve/main/dataset_infos.json"
    data = json.load(urllib.request.urlopen(url, timeout=30))
    names = next(iter(data.values()))["features"]["label"]["names"]
    assert len(names) == 77, f"expected 77 official intents, got {len(names)}"
    return names


def write_jsonl(rows, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def convert(rows, names):
    # one case per query: a single choice question with ALL 77 intents
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
    import random

    names = intent_names()
    assert len(names) == 77, f"expected 77 intents, got {len(names)}"
    train, test = download_banking77(names)

    train_rows = list(convert(train, names))
    # the source csv is sorted by category, so a naive tail-carve produces a
    # valid split holding only the alphabetically-last intents (8 of 77 seen
    # in one bad split). Shuffle with a fixed seed BEFORE carving: valid must
    # span the intent space or valid_loss is meaningless as a model gate.
    random.Random(20260929).shuffle(train_rows)
    valid_rows = train_rows[-1000:]
    seen = {r["questions"][0]["answer"] for r in valid_rows}
    assert len(seen) >= 50, f"valid split covers only {len(seen)}/77 intents: split is broken"
    write_jsonl(train_rows[:-1000], "data/banking77/train.jsonl")
    write_jsonl(valid_rows, "data/banking77/valid.jsonl")
    test_rows = list(convert(test, names))
    write_jsonl(test_rows, "data/banking77/test.jsonl")
    print(f"banking77 train/valid/test written: {len(train_rows) - 1000}/{1000}/{len(test_rows)} cases "
          f"(77 options each, valid spans {len(seen)} intents)")
