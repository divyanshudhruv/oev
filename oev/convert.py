import json
from pathlib import Path

AG_LABELS = ["world", "sports", "business", "sci/tech"]
EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def convert_rows(rows, labels, domain, qname, instructions):
    for i, r in enumerate(rows):
        yield {
            "id": f"{domain}-{i:06d}",
            "domain": domain,
            "state": r["text"],
            "questions": [
                {
                    "name": qname,
                    "type": "choice",
                    "options": list(labels),
                    "answer": labels[r["label"]],
                    "instructions": instructions,
                }
            ],
        }


def download_ag_news():
    from datasets import load_dataset

    return load_dataset("fancyzhx/ag_news")


def download_emotion():
    from datasets import load_dataset

    return load_dataset("dair-ai/emotion")


def build_domain(hf_split_rows, labels, domain, qname, instructions):
    return list(convert_rows(hf_split_rows, labels, domain, qname, instructions))


def write_jsonl(rows, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    ag = download_ag_news()
    em = download_emotion()
    ag_train = build_domain(ag["train"], AG_LABELS, "ag_news", "topic", "Which topic does this news article belong to?")
    write_jsonl(ag_train[:-2000], "data/ag_news/train.jsonl")
    write_jsonl(ag_train[-2000:], "data/ag_news/valid.jsonl")
    write_jsonl(build_domain(ag["test"], AG_LABELS, "ag_news", "topic", "Which topic does this news article belong to?"), "data/ag_news/test.jsonl")
    em_train = build_domain(em["train"], EMOTION_LABELS, "emotion", "emotion", "Which emotion does this text express?")
    write_jsonl(em_train[:-1000], "data/emotion/train.jsonl")
    write_jsonl(em_train[-1000:], "data/emotion/valid.jsonl")
    write_jsonl(build_domain(em["test"], EMOTION_LABELS, "emotion", "emotion", "Which emotion does this text express?"), "data/emotion/test.jsonl")
    print("ag_news train/valid/test written; emotion train/valid/test written")
