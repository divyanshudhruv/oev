# Demo states and question schemas, verbatim from the typed-decisions
# training data. Editing these is safe: they are plain JSON payloads.
import json

AGENT_STATE = ('{"agent": {"autonomy": "checkpointed", "model": "internal-agent-v4"}, '
               '"constraints": ["Do not touch customer data outside the named accounts"], '
               '"task": "Delete personal data for the accounts in the erasure queue.", '
               '"trace_summary": {"constraint_violations": 0, "duration_s": 128.5, '
               '"irreversible_actions": 0, "steps": 7, "tool_errors": 1}}')

AGENT_QS = {
    "action": {"type": "choice", "options": ["continue", "human_review", "observe", "stop"],
               "instructions": "action"},
    "needs_review": {"type": "noul"},
    "outcome": {"type": "choice", "options": ["failure", "harmful", "partial", "success"],
                "instructions": "outcome"},
    "risk": {"type": "score",
             "levels": ["Benign: read-only or clearly safe actions.", "Low: routine writes within scope.",
                        "Moderate: irreversible or out-of-scope actions.",
                        "High: destructive, security-relevant, or policy-violating actions."]},
}


def customer_state(tier, tickets, msg):
    return json.dumps({
        "account": {"lifetime_value_usd": 4642, "prior_tickets_90d": int(tickets),
                    "seats": 3, "tenure_months": 60, "tier": tier},
        "thread": [{"role": "customer", "text": msg}],
    })


CUSTOMER_QS = {
    "action": {"type": "choice",
               "options": ["answer_directly", "close_no_action", "escalate_to_human",
                           "execute_refund", "request_information"],
               "instructions": "action"},
    "category": {"type": "choice", "options": ["account", "billing", "delivery", "refund", "technical"],
                 "instructions": "category"},
    "churn_risk": {"type": "score",
                   "levels": ["No sign of dissatisfaction.", "Mild frustration, but the relationship is intact.",
                              "Clearly unhappy; repeat problems or explicit complaints.",
                              "Imminent: threatening to cancel, dispute or leave."]},
    "needs_human": {"type": "noul"},
}

INVOICE_QS = {
    "discrepancy_severity": {"type": "score",
                             "levels": ["None: everything reconciles.",
                                        "Trivial: rounding or a cosmetic difference.",
                                        "Moderate: a real difference worth confirming.",
                                        "Material: a large or unexplained difference."]},
    "disposition": {"type": "choice", "options": ["approve", "hold", "manual_review", "reject"],
                    "instructions": "disposition"},
    "duplicate": {"type": "noul"},
    "matches_order": {"type": "noul"},
}


def invoice_state(vendor, inv, po, recv, ordered, disputes):
    return json.dumps({
        "delivery": {"condition": "accepted with exceptions", "date": "2026-03-18",
                     "received_qty": int(recv)},
        "invoice": {"currency": "USD", "id": "INV-2026-7551",
                    "lines": [{"qty": int(ordered), "sku": "SKU-940",
                               "total_usd": float(inv),
                               "unit_usd": round(float(inv) / max(int(ordered), 1), 2)}],
                    "total_usd": float(inv), "vendor": vendor},
        "payment": {"days_until_due": -3, "discount_expires_in_days": None,
                    "early_payment_discount_pct": None},
        "purchase_order": {"id": "PO-2026-4102",
                           "lines": [{"qty": int(ordered), "sku": "SKU-940",
                                      "total_usd": float(po),
                                      "unit_usd": round(float(po) / max(int(ordered), 1), 2)}]},
        "vendor_history": {"invoices_last_12m": 14, "disputes_last_12m": int(disputes)},
    })


SECURITY_QS = {
    "credential_compromise": {"type": "noul"},
    "disposition": {"type": "choice", "options": ["close_benign", "contain", "investigate", "monitor"],
                    "instructions": "disposition"},
    "severity": {"type": "score",
                 "levels": ["Negligible: no access to anything sensitive.",
                            "Low: limited access, easily reversed.",
                            "Moderate: access to internal systems or non-public data.",
                            "High: access to production, secrets or customer data.",
                            "Critical: active compromise of crown-jewel systems."]},
    "true_positive": {"type": "noul"},
}


def security_state(rule, desc, evid, crit, ptype):
    return json.dumps({
        "alert": {"description": desc, "evidence": evid, "rule": rule},
        "context": {"asset_criticality": crit, "change_window_active": False},
        "history": {"prior_incidents_90d": 0},
        "principal": {"department": "engineering", "roles": ["service"], "type": ptype},
    })


# preset builders pass ready-made args; these two carry the story the demo
# should tell: a real invoice discrepancy and a genuinely suspicious login
def invoice_state_discrepant():
    # invoice overbills 10% vs the PO on the same qty and sku: a material,
    # unambiguous mismatch the generalist should call out
    return invoice_state("Acme Fabrication", 330022.0, 300020.0, 1000, 1000, 0)


SUSPICIOUS_LOGIN = (
    "dormant_account_reactivation",
    "a dormant privileged service account resumed production access",
    ("The service account `svc_deploy_prod` with read-write access to the "
     "payments database initiated a session from IP `198.51.100.24`, a range "
     "never seen for this account. Authentication succeeded without MFA using "
     "credentials last rotated 400 days ago. The session immediately ran a "
     "bulk export against the customer_payments table at 03:12 local time, "
     "outside any change window."),
    "high",
    "service_account",
)


PLAYGROUND_STATE = "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card. What are you going to do about this?"

PLAYGROUND_QS = {
    "department": {"type": "choice",
                   "instructions": "Which team should handle this?",
                   "options": ["billing", "shipping", "other"]},
    "escalate": {"type": "noul", "instructions": "Does this need urgent human attention?"},
    "frustration": {"type": "score", "levels": ["calm", "frustrated", "very angry"]},
}

# cross-field sanity rules: (if_field, if_value, then_field, expected)
COHERENCE_RULES = [
    ("action", "human_review", "needs_review", "yes"),
    ("action", "stop", "needs_review", "yes"),
    ("action", "continue", "needs_review", "no"),
]
