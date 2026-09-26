import pytest

from oev.infer import OEV, normalize_question


def test_decide_outputs(trained_checkpoint):
    agent = OEV(trained_checkpoint)
    out = agent.decide(
        "We were charged twice and need a refund urgently.",
        {
            "department": {"type": "choice", "options": ["billing", "technical", "other"]},
            "urgent": {"type": "noul"},
            "severity": {"type": "score", "levels": [1, 2, 3, 4, 5]},
        },
    )
    assert set(out) == {"department", "urgent", "severity"}
    assert abs(sum(out["department"]["probabilities"].values()) - 1.0) < 1e-5
    assert 0.0 <= out["urgent"] <= 1.0
    assert out["severity"]["value"] in (1, 2, 3, 4, 5)
    assert out["department"]["choice"] in ("billing", "technical", "other")


def test_decide_accepts_jev_criteria_schema(trained_checkpoint):
    agent = OEV(trained_checkpoint)
    out = agent.decide(
        "We were charged twice and need a refund urgently.",
        {
            "department": {
                "type": "choice",
                "instructions": "Which department?",
                "criteria": {"billing": "payments and refunds", "technical": "bugs and outages", "other": "everything else"},
            },
            "urgent": {"type": "noul", "criteria": {"true": "blocks automation", "false": "routine"}},
            "severity": {"type": "score", "instructions": "How bad?", "criteria": ["minor", "serious", "critical"]},
        },
    )
    assert out["department"]["choice"] in ("billing", "technical", "other")
    assert set(out["department"]["probabilities"]) == {"billing", "technical", "other"}
    assert 0.0 <= out["urgent"] <= 1.0
    assert out["severity"]["value"] in ("minor", "serious", "critical")
    assert set(out["severity"]["probabilities"]) == {"minor", "serious", "critical"}


def test_normalize_question_rejects_missing_options():
    with pytest.raises(ValueError):
        normalize_question("x", {"type": "choice"})
    with pytest.raises(ValueError):
        normalize_question("x", {"type": "score"})


def test_normalize_question_prefers_native_keys():
    q = normalize_question("x", {"type": "choice", "options": ["a"], "criteria": {"b": "ignored"}})
    assert q["options"] == ["a"]
    q = normalize_question("x", {"type": "score", "levels": [1, 2], "criteria": ["a", "b"]})
    assert q["levels"] == [1, 2]


def test_decide_temperature_argument_does_not_mutate_agent(trained_checkpoint):
    agent = OEV(trained_checkpoint)
    original_temperature = agent.temperature

    agent.decide(
        "We need a refund.",
        {"urgent": {"type": "noul"}},
        temperature=2.0,
    )

    assert agent.temperature == original_temperature
