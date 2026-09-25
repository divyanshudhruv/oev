import json
from pathlib import Path

NOUL_OPTIONS = ["no", "yes"]


def render_state(raw):
    try:
        obj = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return str(raw)
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return "\n".join(f"{k}: {v}" for k, v in obj.items())
    return str(obj)


def score_levels(gold_q, criteria):
    probs = gold_q.get("probabilities", {})
    int_levels = [int(k) for k in probs if str(k).lstrip("-").isdigit()]
    n = len(criteria) if isinstance(criteria, list) else 4
    label = int(gold_q["label"])
    return max(n, label + 1, (max(int_levels) + 1 if int_levels else 0))


def soft_target(gold_q, options):
    probs = gold_q.get("probabilities", {})
    total = 0.0
    target = []
    for o in options:
        p = probs.get(o)
        if p is None and o == "no":
            p = probs.get("false")
        elif p is None and o == "yes":
            p = probs.get("true")
        p = float(p) if p is not None else 0.0
        target.append(p)
        total += p
    if total <= 0.0:
        return None
    return [p / total for p in target]


def _enrich(option, desc):
    d = (desc or "").strip()
    return f"{option}: {d}" if d else option


def convert_question(qname, q, gold_q):
    t = q["type"]
    crit = q.get("criteria", {})
    base = {"name": qname, "type": t, "instructions": q.get("instructions", t)}
    if t == "choice":
        keys = list(crit.keys())
        if str(gold_q["label"]) not in keys:
            return None
        target = soft_target(gold_q, keys)
        options = [_enrich(k, crit[k]) for k in keys]
        answer = options[keys.index(str(gold_q["label"]))]
    elif t == "noul":
        options = NOUL_OPTIONS
        target = soft_target(gold_q, options)
        answer = "yes" if str(gold_q["label"]).lower() == "true" else "no"
    elif t == "score":
        n = score_levels(gold_q, crit)
        raw = [str(i) for i in range(n)]
        target = soft_target(gold_q, raw)
        descs = crit if isinstance(crit, list) else []
        options = [_enrich(str(i), descs[i] if i < len(descs) else "") for i in range(n)]
        label = int(gold_q["label"])
        if label >= n:
            return None
        answer = options[label]
    else:
        return None
    cq = {**base, "options": options, "answer": answer}
    if target is not None:
        cq["target"] = target
    return cq


def convert_rows_typed(rows):
    for row in rows:
        questions = json.loads(row["questions"]) if isinstance(row["questions"], str) else row["questions"]
        gold = json.loads(row["gold"]) if isinstance(row["gold"], str) else row["gold"]
        qds = []
        for qname, q in questions.items():
            if qname not in gold:
                continue
            cq = convert_question(qname, q, gold[qname])
            if cq is not None:
                qds.append(cq)
        if not qds:
            continue
        yield {
            "id": f"typed-{row['id']}",
            "domain": row.get("workflow", "typed"),
            "state": render_state(row["state"]),
            "questions": qds,
        }


def download_typed():
    from datasets import load_dataset

    return load_dataset("LocalLLaMA/typed-decisions", "all")


def write_jsonl(rows, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


if __name__ == "__main__":
    ds = download_typed()
    train_rows = list(convert_rows_typed(ds["train"]))
    write_jsonl(train_rows[:-200], "data/typed/train.jsonl")
    write_jsonl(train_rows[-200:], "data/typed/valid.jsonl")
    write_jsonl(list(convert_rows_typed(ds["test"])), "data/typed/test.jsonl")
    print(f"typed train/valid/test written: {len(train_rows) - 200}/{200}/{len(ds['test'])} cases")
