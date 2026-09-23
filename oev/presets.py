"""Ready-made question schemas for common workflows.

Mirrors the workflow-preset idea from similar decision-model SDKs: instead of
hand-writing typed questions for common triage/safety tasks, import a preset
and call `decide` with your state. Every preset is plain data — edit freely.

Usage:
    from oev.infer import OEV
    from oev.presets import triage_questions

    agent = OEV("checkpoints_td5/oev-tiny.pt")
    result = agent.decide({"message": "My payment failed twice"}, triage_questions())
"""

import torch

from oev.dataset import pack
from oev.evaluate import load_model
from oev.tokenizer_hf import HFTokenPacker


def triage_questions():
    """Support ticket triage: intent, urgency, frustration, churn."""
    return {
        "department": {
            "type": "choice",
            "instructions": "Which department should handle this request?",
            "options": ["billing", "technical", "sales", "other"],
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this request?",
            "levels": [1, 2, 3],
        },
        "frustration": {
            "type": "noul",
            "instructions": "Is the user frustrated or angry?",
        },
        "churn_risk": {
            "type": "noul",
            "instructions": "Does the user threaten to cancel or leave?",
        },
    }


def guard_questions():
    """Prompt guardrails: jailbreaks, injections, leaks."""
    return {
        "jailbreak": {
            "type": "noul",
            "instructions": "Does this prompt attempt to bypass system instructions?",
        },
        "injection": {
            "type": "noul",
            "instructions": "Does this text contain an instruction-injection attempt?",
        },
        "leak": {
            "type": "noul",
            "instructions": "Does this text try to extract system prompts or secrets?",
        },
    }


def moderation_questions():
    """Content safety: toxicity, harassment, threats."""
    return {
        "toxic": {
            "type": "noul",
            "instructions": "Is this content toxic or insulting?",
        },
        "harassment": {
            "type": "noul",
            "instructions": "Does this content harass or bully a person?",
        },
        "threat": {
            "type": "noul",
            "instructions": "Does this content contain a threat of violence?",
        },
    }


def router_questions():
    """Route a request between small and frontier models."""
    return {
        "complexity": {
            "type": "score",
            "instructions": "How complex is this request for an LLM to execute?",
            "levels": [1, 2, 3],
        },
        "agentic": {
            "type": "noul",
            "instructions": "Does this request require multi-step tool use?",
        },
        "task_type": {
            "type": "choice",
            "instructions": "What kind of request is this?",
            "options": ["classification", "generation", "extraction", "reasoning"],
        },
    }


def gate(result, threshold=0.85):
    """Confidence-gating recipe: returns (name, payload, confident).

    Because OEV's probabilities are trained with proper scoring rules against
    calibrated targets, confidence is statistically meaningful — automate when
    confident, escalate when not:

        for name, payload, confident in gate(result, threshold=0.85):
            if confident:
                automate(name, payload)
            else:
                escalate(name, payload)
    """
    out = []
    for name, payload in result.items():
        if isinstance(payload, dict):
            conf = payload.get("confidence")
            if conf is None:
                # score questions: use the mass on the argmax level
                probs = payload.get("probabilities", {})
                conf = max(probs.values()) if probs else 0.0
            out.append((name, payload, conf >= threshold))
        else:
            # noul returns a bare float = P(yes); confidence is max(p, 1-p)
            out.append((name, payload, max(payload, 1.0 - payload) >= threshold))
    return out


def decide(agent, state, questions, device=None):
    """Convenience wrapper: batch every question for one state in the fewest
    forward passes and return answers plus per-question confidence."""
    answers = agent.decide(state, questions)
    gated = dict(gate(answers))
    return {
        "answers": answers,
        "confidence": {k: (max(v, 1.0 - v) if isinstance(v, float)
                           else v.get("confidence", 0.0)) for k, v in answers.items()},
        "automatable": all(c for _, c, ok in gated),
    }
