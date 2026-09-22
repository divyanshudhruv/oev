from oev.data_gen import generate


def test_deterministic():
    assert generate(50, seed=7) == generate(50, seed=7)


def test_split_disjoint_states():
    tr, va, te = generate(120, seed=3)
    a = {r["id"] for r in tr}
    b = {r["id"] for r in va}
    c = {r["id"] for r in te}
    assert not (a & b) and not (a & c) and not (b & c)
    assert len(tr) + len(va) + len(te) == 120


def test_answers_are_valid_strings():
    _, _, te = generate(90, seed=5)
    for r in te:
        for q in r["questions"]:
            assert q["answer"] in q["options"]
            assert q["type"] in ("choice", "noul", "score")


def test_domains_covered():
    tr, _, _ = generate(90, seed=11)
    assert {r["domain"] for r in tr} == {"support", "agent_trace", "review"}
