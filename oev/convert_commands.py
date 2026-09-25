"""Synthetic command-intent dataset: the OEV Live router specialist.

Generates voice-command transcripts labeled with an intent choice question and
a completeness noul question. The completeness signal is what powers the
observe / prefetch / commit pipeline: a partial transcript ("open spotify and
play...") is intent=unknown or low-confidence intent with complete=no, while a
finished command is complete=yes. A checkpoint fine-tuned on this split becomes
the decision model for the real-time command router demo.

Run:
    python -m oev.convert_commands             # writes data/commands/{train,valid,test}.jsonl
    python -m oev.benchmark_ext --checkpoint <router.pt> --data-dir data/commands

The data is fully synthetic (seeded, reproducible); no download required.
"""
import json
import os
import random

INTENTS = ["music", "search", "note", "calculator", "app", "unknown"]

INSTRUCTIONS_INTENT = ("What does the user want the computer to do? "
                       "unknown covers partial commands, bare triggers and chat.")

APPS = ["spotify", "chrome", "notepad", "calculator", "vs code", "discord",
        "steam", "explorer", "settings", "terminal"]

SONGS = ["blinding lights", "bohemian rhapsody", "take five", "gimme shelter",
         "fly me to the moon", "dream on", "landslide", "redbone",
         "electric feel", "ain't no mountain high enough"]

ARTISTS = ["the weeknd", "queen", "dave brubeck", "nina simone", "daft punk",
           "fleetwood mac", "tame impala", "miles davis", "chaka khan"]

QUERIES = ["rust async programming", "latest nvidia drivers", "weather tomorrow",
           "best noise cancelling headphones", "python deque vs list",
           "how to fold a fitted sheet", "distance to the moon",
           "cheap flights to lisbon", "markdown table syntax", "is pluto a planet"]

TASKS = ["finish the benchmark writeup", "call the dentist", "water the plants",
         "back up the laptop", "renew the domain", "ship the release notes"]

ITEMS = ["oat milk", "printer paper", "a new mouse", "birthday candles", "coffee beans"]
DAYS = ["saturday", "monday", "friday", "tomorrow"]
PEOPLE = ["sam", "priya", "marco", "lena"]

CHAT = ["hello", "hey there", "good morning", "how are you", "thanks a lot",
        "what time is it", "tell me a joke", "never mind", "actually stop"]

MUSIC_T = ["play {song} on spotify", "open spotify and play {song}",
           "put on {artist}", "play some {artist} songs", "queue up {song}",
           "play {song} by {artist}", "start {artist} radio"]
SEARCH_T = ["search the web for {query}", "search youtube for {query}",
            "google {query}", "look up {query}", "find {query} on the web"]
NOTE_T = ["write in my notes that I need to {task}", "take a note: {task}",
          "remind me to {task}", "note that I have to {task}"]
CALC_T = ["calculate {a} times {b}", "what is {a} plus {b}", "what is {a} minus {b}",
          "calculate {a} divided by {b}", "{a} times {b}"]
APP_T = ["open {app}", "launch {app}", "start {app}", "switch to {app}",
         "open up {app}"]


def _fill(templates, rng, **pools):
    t = rng.choice(templates)
    return t.format(**{k: rng.choice(v) for k, v in pools.items()})


def _full_command(rng):
    kind = rng.choice(["music", "search", "note", "calculator", "app"])
    if kind == "music":
        return _fill(MUSIC_T, rng, song=SONGS, artist=ARTISTS), "music"
    if kind == "search":
        return _fill(SEARCH_T, rng, query=QUERIES), "search"
    if kind == "note":
        task = rng.choice(TASKS) + (" " + rng.choice(["today", "this week", "by friday"])
                                     if rng.random() < 0.4 else "")
        return rng.choice(NOTE_T).format(task=task), "note"
    if kind == "calculator":
        a, b = rng.randint(2, 999), rng.randint(2, 99)
        return _fill(CALC_T, rng, a=[a], b=[b]), "calculator"
    return _fill(APP_T, rng, app=APPS), "app"


def _partial(full, rng):
    words = full.split()
    cut = rng.randint(1, max(1, len(words) - 2))
    return " ".join(words[:cut])


def _case(i, state, intent, complete, prefix):
    return {
        "id": f"{prefix}-{i:06d}",
        "domain": "commands",
        "state": state,
        "questions": [
            {"name": "intent", "type": "choice", "instructions": INSTRUCTIONS_INTENT,
             "options": INTENTS, "answer": intent},
            {"name": "complete", "type": "noul",
             "instructions": "Is this a finished command, or is the user still speaking?",
             "answer": "yes" if complete else "no"},
        ],
    }


def generate(n, seed, prefix):
    """Balanced generation: intent classes get equal quotas; within each class
    the full/partial/trigger/chat mix teaches the observe/prefetch/commit split.
    Repeats (unavoidable for small pools like apps) are re-randomized instead of
    being dropped, so a class is never starved by dedupe."""
    rng = random.Random(seed)
    # one slot per case, cycling intent classes so every class gets n/len(INTENTS)
    slots = [INTENTS[i % len(INTENTS)] for i in range(n)]
    rng.shuffle(slots)
    out = []
    for state_slot in slots:
        r = rng.random()
        if state_slot == "unknown":
            # unknown = partials, bare triggers, chat (never full commands)
            if r < 0.55:
                full, _ = _full_command(rng)
                state = _partial(full, rng)
            elif r < 0.8:
                state = rng.choice(["open", "play", "search", "write", "calculate",
                                    "can you", "hey", "um", "launch"])
            else:
                state = rng.choice(CHAT)
            out.append(_case(len(out), state, "unknown", rng.random() < 0.3, prefix))
            continue
        # a real intent: 60% full command, 25% partial, 15% bare trigger of that intent
        full, _ = _full_command(rng)
        # force the template family to match the slot
        for _ in range(50):
            full, kind = _full_command(rng)
            if kind == state_slot:
                break
        if r < 0.6:
            out.append(_case(len(out), full, state_slot, True, prefix))
        elif r < 0.85:
            out.append(_case(len(out), _partial(full, rng), state_slot, False, prefix))
        else:
            trigger = full.split()[0]
            out.append(_case(len(out), trigger, state_slot, False, prefix))
    return out


def main(out_dir="data/commands"):
    os.makedirs(out_dir, exist_ok=True)
    sizes = {"train": 8000, "valid": 1000, "test": 1000}
    seeds = {"train": 11, "valid": 22, "test": 33}
    for split, n in sizes.items():
        path = os.path.join(out_dir, f"{split}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for c in generate(n, seeds[split], split[:2]):
                f.write(json.dumps(c) + "\n")
        print(f"wrote {n} cases to {path} (intents: {', '.join(INTENTS)})")


if __name__ == "__main__":
    main()
