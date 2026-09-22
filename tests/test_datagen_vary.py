from oev.data_gen import generate, vary_state


def test_variation_preserves_answer_relevant_words():
    s = {"domain": "review", "state": "Review of headphones: it was terrible.", "questions": [{"name": "sentiment", "type": "choice", "options": ["positive", "negative", "neutral"], "answer": "negative"}]}
    v = vary_state(s)
    assert "terrible" in v["state"].lower()


def test_variation_changes_surface_text():
    s = {"domain": "support", "state": "Subject: question about invoice. We were charged twice for the same order.", "questions": [{"name": "urgent", "type": "noul", "options": ["no", "yes"], "answer": "no"}]}
    variants = {vary_state(s, seed_extra=str(k))["state"] for k in range(30)}
    assert len(variants) > 5


def test_generate_varied_flag_changes_states_not_answers():
    plain = generate(60, seed=4)
    varied = generate(60, seed=4, vary=True)
    assert plain[0][0]["questions"] == varied[0][0]["questions"]
    assert any(a["state"] != b["state"] for a, b in zip(plain[0], varied[0]))
