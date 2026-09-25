import importlib
import json
import sys
from pathlib import Path

import gradio as gr
import pytest


@pytest.fixture
def app_module(monkeypatch):
    launches = []
    monkeypatch.setattr(gr.Blocks, "launch", lambda self, *args, **kwargs: launches.append(True))
    sys.modules.pop("app", None)
    module = importlib.import_module("app")
    monkeypatch.setattr(module, "_test_launches", launches, raising=False)
    return module


def test_noul_winner_follows_probability(app_module):
    payload, markdown = app_module._render_result(
        {"review": 0.2},
        {"review": {"type": "noul"}},
    )

    assert payload["review"]["answer"] == "no"
    assert 'class="row win"' in markdown
    assert 'name">no<' in markdown
    assert 'class="row win"><span class="name">yes"' not in markdown


def test_render_escapes_dynamic_labels(app_module):
    label = '<img src=x onerror="alert(1)">'
    _, markdown = app_module._render_result(
        {"choice": {"choice": label, "probabilities": {label: 1.0}}},
        {"choice": {"type": "choice", "options": [label]}},
    )

    assert "<img" not in markdown
    assert "&lt;img" in markdown


def test_safe_probability_clamps_invalid_values(app_module):
    assert app_module._safe_probability(-1.0) == 0.0
    assert app_module._safe_probability(2.0) == 1.0
    assert app_module._safe_probability(float("nan")) == 0.0


def test_validate_rejects_duplicate_options(app_module):
    questions, error = app_module._validate(
        "state",
        json.dumps({"choice": {"type": "choice", "options": ["same", "same"]}}),
    )

    assert questions is None
    assert "unique" in error


def test_validate_accepts_numeric_score_levels(app_module):
    questions, error = app_module._validate(
        "state",
        json.dumps({"severity": {"type": "score", "levels": [1, 2, 3]}}),
    )

    assert error is None
    assert questions["severity"]["levels"] == [1, 2, 3]


def test_validate_rejects_equivalent_score_levels(app_module):
    questions, error = app_module._validate(
        "state",
        json.dumps({"severity": {"type": "score", "levels": [1, "1"]}}),
    )

    assert questions is None
    assert "unique" in error


def test_import_does_not_launch_demo(app_module):
    assert app_module._test_launches == []


def test_run_does_not_leave_temperature_changed_after_error(app_module, monkeypatch):
    class FailingAgent:
        temperature = 1.0

        def decide(self, state, questions, temperature=None):
            raise RuntimeError("boom")

    agent = FailingAgent()
    monkeypatch.setattr(app_module, "load", lambda: agent)

    with pytest.raises(RuntimeError):
        app_module._run("state", {"review": {"type": "noul"}}, 2.0)

    assert agent.temperature == 1.0


def test_ui_uses_warm_dark_pastel_theme_and_normalizes_text(app_module):
    css = app_module.CSS
    source = Path(app_module.__file__).read_text(encoding="utf-8")
    assert "primary_hue=\"orange\"" in source
    assert "body_background_fill=\"#171412\"" in source
    assert "body_text_color=\"#f5eee8\"" in source
    assert "block_background_fill=\"#211d1a\"" in source
    assert "color_accent=\"#e8b48c\"" in source
    assert " color_accent_dark=\"" not in source
    assert "--primary-pastel: #e8b48c" in css
    assert "font-size: 14px" in css
    assert "line-height: 1.5" in css
    assert "font-size: 12px" in css
    assert "font-size: 13px" in css
    assert "0.86em" not in css


def test_ui_keeps_desktop_shell_stable_and_responsive(app_module):
    css = app_module.CSS
    assert "width: 1020px" in css
    assert "min-width: 0" in css
    assert "gap: 16px" in css
    assert "margin-bottom: 16px" in css
    assert ".tab-container" in css
    assert ".tab-container .column { width: 100%" not in css
    assert "@media (max-width: 640px)" in css
