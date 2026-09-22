from oev.infer import OEV


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
