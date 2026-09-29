# Ready-made question schemas for common workflows. Plain data: import one,
# pass your state to agent.decide, edit freely.




def triage_questions():
    # support ticket triage: intent, urgency, frustration, churn
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
    # prompt guardrails: jailbreaks, injections, leaks
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
    # content safety: toxicity, harassment, threats
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
    # route a request between small and frontier models
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
    # (name, payload, confident) per answer: automate when confident,
    # escalate when not. Confidence is trained against calibrated targets,
    # so the threshold is statistically meaningful.
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
    # answers plus per-question confidence and an overall automatable flag
    answers = agent.decide(state, questions)
    gated = dict(gate(answers))
    return {
        "answers": answers,
        "confidence": {k: (max(v, 1.0 - v) if isinstance(v, float)
                           else v.get("confidence", 0.0)) for k, v in answers.items()},
        "automatable": all(c for _, c, ok in gated),
    }
