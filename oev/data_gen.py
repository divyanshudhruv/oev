import json
import random
from pathlib import Path

DEPARTMENTS = ["billing", "technical", "sales", "other"]
ACTIONS = ["read_file", "edit_file", "run_tests", "search", "retry", "stop"]
SENTIMENTS = ["positive", "negative", "neutral"]

OPENERS = ["", "Hello, ", "Hi team, ", "To whom it may concern: ", "Quick one - "]
URGENT_PHRASES = ["This is blocking us.", "Please handle this today.", "We need an answer now.", "This is an emergency for our launch."]
CALM_PHRASES = ["No rush on this.", "Whenever you get a chance.", "This can wait until next week.", "Just a general question."]
REVIEW_PREFIX = ["", "Bought this last month. ", "Used it for two weeks. ", "Second one I purchased. "]
AGENT_OPENERS = ["Agent state:", "Current run:", "Trace snapshot:", "Run status:"]


def vary_state(s, seed_extra=""):
    import copy
    import re

    rng = random.Random(hash(s.get("id", "") + seed_extra + s["state"][:50]) & 0xFFFFFFFF)
    out = copy.deepcopy(s)
    text = out["state"]
    text = rng.choice(OPENERS) + text
    if out["domain"] == "support":
        text = re.sub(r"^Subject: question about ", "Subject: ", text)
        if any(w in text.lower() for w in ("charged twice", "refund", "frustrating", "fails with an error")):
            if rng.random() < 0.4:
                text += " " + rng.choice(URGENT_PHRASES)
        elif rng.random() < 0.4:
            text += " " + rng.choice(CALM_PHRASES)
    elif out["domain"] == "agent_trace":
        text = text.replace("Agent state:", rng.choice(AGENT_OPENERS))
    elif out["domain"] == "review":
        text = rng.choice(REVIEW_PREFIX) + text
        text = text.replace("it was ", "it felt ") if rng.random() < 0.5 else text
    out["state"] = text
    return out


def support_state(rng):
    topic = rng.choice(["invoice", "payment", "login", "bug", "pricing", "feature"])
    charged_twice = rng.random() < 0.3 and topic in ("invoice", "payment")
    refund_words = rng.random() < 0.5 and topic in ("invoice", "payment")
    broken = topic in ("login", "bug")
    angry = rng.random() < 0.35
    deadline = rng.random() < 0.3
    text = f"Subject: question about {topic}. "
    if charged_twice:
        text += "We were charged twice for the same order. "
    if refund_words:
        text += "Please refund the duplicate charge to our card. "
    if broken:
        text += f"The {topic} page fails with an error since yesterday. "
    if angry:
        text += "This is very frustrating, fix it now. "
    if deadline:
        text += "We have a critical launch deadline tomorrow. "
    facts = {
        "charged_twice": charged_twice,
        "refund_words": refund_words,
        "broken": broken,
        "angry": angry,
        "deadline": deadline,
        "topic": topic,
    }
    return {"domain": "support", "state": text.strip(), "facts": facts}


def support_questions(facts):
    if facts["charged_twice"] or facts["refund_words"]:
        dept = "billing"
    elif facts["broken"]:
        dept = "technical"
    elif facts["topic"] == "pricing":
        dept = "sales"
    else:
        dept = "other"
    severity = min(5, 1 + (2 if facts["broken"] else 0) + (2 if facts["deadline"] else 0) + (1 if facts["angry"] else 0))
    urgent = facts["deadline"] or facts["angry"]
    refund = facts["refund_words"] or facts["charged_twice"]
    return [
        {"name": "department", "type": "choice", "options": DEPARTMENTS, "answer": dept},
        {"name": "urgent", "type": "noul", "options": ["no", "yes"], "answer": "yes" if urgent else "no"},
        {"name": "refund_requested", "type": "noul", "options": ["no", "yes"], "answer": "yes" if refund else "no"},
        {"name": "severity", "type": "score", "options": ["1", "2", "3", "4", "5"], "answer": str(severity)},
    ]


def agent_state(rng):
    tests_failed = rng.random() < 0.4
    files_changed = rng.randint(0, 5)
    attempts = rng.randint(0, 3)
    error = rng.choice(["none", "typescript", "network", "assertion"])
    done = (not tests_failed) and error == "none" and rng.random() < 0.5
    text = (
        f"Agent state: files_changed={files_changed}, attempts={attempts}, "
        f"last_error={error}. Tests {'failed' if tests_failed else 'passed'}. "
        f"Task complete: {str(done).lower()}."
    )
    facts = {
        "tests_failed": tests_failed,
        "files_changed": files_changed,
        "attempts": attempts,
        "error": error,
        "done": done,
    }
    return {"domain": "agent_trace", "state": text, "facts": facts}


def agent_questions(facts):
    if facts["done"]:
        action = "stop"
    elif facts["error"] == "network" and facts["attempts"] < 3:
        action = "retry"
    elif facts["tests_failed"] or facts["error"] in ("typescript", "assertion"):
        action = "edit_file"
    else:
        action = "read_file"
    progress = 5 if facts["done"] else (2 if facts["tests_failed"] else 3)
    return [
        {"name": "next_action", "type": "choice", "options": ACTIONS, "answer": action},
        {"name": "should_stop", "type": "noul", "options": ["no", "yes"], "answer": "yes" if facts["done"] else "no"},
        {"name": "progress", "type": "score", "options": ["1", "2", "3", "4", "5"], "answer": str(progress)},
    ]


POS_WORDS = ["great", "love", "excellent", "amazing"]
NEG_WORDS = ["terrible", "broke", "waste", "awful"]
NEU_WORDS = ["average", "okay", "standard", "fine"]


def review_state(rng):
    rating = rng.randint(1, 5)
    if rating >= 4:
        word = rng.choice(POS_WORDS)
    elif rating <= 2:
        word = rng.choice(NEG_WORDS)
    else:
        word = rng.choice(NEU_WORDS)
    product = rng.choice(["headphones", "keyboard", "charger", "monitor"])
    text = f"Review of {product}: it was {word}. "
    if rating <= 2:
        text += "Stopped working after a few days. "
    if rating >= 4:
        text += "Would buy it again. "
    return {"domain": "review", "state": text.strip(), "facts": {"rating": rating}}


def review_questions(facts):
    rating = facts["rating"]
    sentiment = "positive" if rating >= 4 else ("negative" if rating <= 2 else "neutral")
    return [
        {"name": "sentiment", "type": "choice", "options": SENTIMENTS, "answer": sentiment},
        {"name": "rating", "type": "score", "options": ["1", "2", "3", "4", "5"], "answer": str(rating)},
        {"name": "recommends", "type": "noul", "options": ["no", "yes"], "answer": "yes" if rating >= 4 else "no"},
    ]


MAKERS = {
    "support": (support_state, support_questions),
    "agent_trace": (agent_state, agent_questions),
    "review": (review_state, review_questions),
}


def generate(n_states, seed, vary=False):
    rng = random.Random(seed)
    names = list(MAKERS)
    states = []
    for i in range(n_states):
        maker, qbuilder = MAKERS[names[i % 3]]
        s = maker(rng)
        s["questions"] = qbuilder(s["facts"])
        del s["facts"]
        s["id"] = f"{s['domain']}-{i:06d}"
        if vary:
            s = vary_state(s, seed_extra=str(i))
        states.append(s)
    order = list(range(n_states))
    random.Random(seed + 1).shuffle(order)
    n_tr = int(n_states * 0.8)
    n_va = int(n_states * 0.1)
    tr = [states[i] for i in order[:n_tr]]
    va = [states[i] for i in order[n_tr : n_tr + n_va]]
    te = [states[i] for i in order[n_tr + n_va :]]
    return tr, va, te


def write_splits(splits, data_dir="data"):
    d = Path(data_dir)
    d.mkdir(parents=True, exist_ok=True)
    for name, rows in zip(["train", "valid", "test"], splits):
        with open(d / f"{name}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)


if __name__ == "__main__":
    write_splits(generate(20000, seed=13, vary=True))
