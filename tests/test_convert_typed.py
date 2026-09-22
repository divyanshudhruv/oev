from oev.convert_typed import convert_question, convert_rows_typed


def fake_rows():
    return [
        {
            "id": "cs-001",
            "workflow": "customer_service",
            "state": '{"from": "user@acme.com", "body": "duplicate charge"}',
            "questions": '{"department": {"type": "choice", "instructions": "Which dept?", "criteria": {"billing": "x", "technical": "y", "other": "z"}}, "churn": {"type": "noul", "instructions": "Will they churn?"}, "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "med", "high"]}}',
            "gold": '{"department": {"label": "billing"}, "churn": {"label": "true"}, "urgency": {"label": 2}}',
        },
        {
            "id": "cs-002",
            "workflow": "invoice",
            "state": "plain text state",
            "questions": '{"weird": {"type": "choice", "criteria": {"a": "x"}}, "bad": {"type": "unknown"}}',
            "gold": '{"weird": {"label": "c"}, "bad": {"label": "x"}}',
        },
    ]


def test_convert_question_types():
    q = {"type": "choice", "criteria": {"a": "x", "b": "y"}}
    gold = {"label": "b"}
    assert convert_question("n", q, gold) == {"name": "n", "type": "choice", "instructions": "choice", "options": ["a", "b"], "answer": "b"}

    gold_n = {"label": "false"}
    assert convert_question("n", {"type": "noul"}, gold_n)["answer"] == "no"

    gold_s = {"label": 1}
    assert convert_question("n", {"type": "score", "criteria": ["l", "m", "h"]}, gold_s)["options"] == ["0", "1", "2"]


def test_rows_shape_and_domain():
    rows = list(convert_rows_typed(fake_rows()))
    assert len(rows) == 1
    r = rows[0]
    assert r["id"] == "typed-cs-001"
    assert r["domain"] == "customer_service"
    assert "from: user@acme.com" in r["state"]
    churn = next(q for q in r["questions"] if q["name"] == "churn")
    assert churn["type"] == "noul" and churn["options"] == ["no", "yes"] and churn["answer"] == "yes"
    u = next(q for q in r["questions"] if q["name"] == "urgency")
    assert u["answer"] == "2"


def test_invalid_questions_skipped():
    rows = list(convert_rows_typed(fake_rows()))
    assert all(q["name"] not in ("weird", "bad") for r in rows for q in r["questions"])
