import pytest

pytest.importorskip("datasets")
from oev.convert import convert_rows

AG_LABELS = ["world", "sports", "business", "sci/tech"]


def fake_ag_news():
    return [
        {"text": "Wall St. Bears Claw Back Into the Black", "label": 2},
        {"text": "Nigeria plans space program", "label": 3},
    ]


def test_choice_shape():
    rows = list(convert_rows(fake_ag_news(), AG_LABELS, "ag_news", "topic", "Which topic does this news article belong to?"))
    assert rows[0]["domain"] == "ag_news"
    assert rows[0]["questions"] == [
        {
            "name": "topic",
            "type": "choice",
            "options": AG_LABELS,
            "answer": "business",
            "instructions": "Which topic does this news article belong to?",
        }
    ]
    assert "Wall St." in rows[0]["state"]


def test_sealed_split_no_overlap():
    tr = [{"id": "a", "domain": "ag_news", "state": "s1", "questions": []}]
    te = [{"id": "b", "domain": "ag_news", "state": "s2", "questions": []}]
    assert {r["id"] for r in tr} & {r["id"] for r in te} == set()
