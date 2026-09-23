"""Worked example: support ticket triage.

Reads a ticket, picks a department, checks whether a refund was requested,
rates severity - then decides whether to automate or escalate.
"""

from oev.infer import OEV
from oev.presets import gate

QUESTIONS = {
    "department": {
        "type": "choice",
        "options": ["billing", "technical", "sales", "other"],
        "instructions": "Which department should handle this ticket?",
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the customer explicitly request a refund?",
    },
    "severity": {"type": "score", "levels": [1, 2, 3, 4, 5],
                 "instructions": "How urgent is this ticket?"},
}

TICKETS = [
    "We were charged twice for the same order. Please refund the duplicate today or we cancel.",
    "The app crashes when I upload a file over 50MB.",
    "Do you have an enterprise plan? We need quotes for 200 seats.",
    "My invoice total looks wrong compared to the order confirmation.",
]


def main():
    agent = OEV("checkpoints_td5/oev-tiny.pt", device="cpu")

    for ticket in TICKETS:
        result = agent.decide(ticket, QUESTIONS)
        print(f"\n{ticket[:60]}...")
        for name, a in result.items():
            if isinstance(a, dict) and "probabilities" in a:
                top = a.get("choice") or a.get("value")
                print(f"  {name}: {top} ({a.get('confidence', 0):.2f})")
            else:
                print(f"  {name}: {a:.2f}")

        # automate only what clears the confidence bar
        actions = gate(result, threshold=0.85)
        for name, payload, confident in actions:
            verdict = "automate" if confident else "route to a human"
            print(f"  -> {name}: {verdict}")


if __name__ == "__main__":
    main()
