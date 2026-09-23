# OEV HF Space demo - simple interface.
# Preset workflows use trained schemas from typed-decisions; the playground
# accepts anything and gets an honest calibrated answer.

import json
import os

import gradio as gr
import torch

try:
    import spaces  # provided by HF ZeroGPU runtime
    _gpu = spaces.GPU
except ImportError:
    _gpu = lambda fn: fn  # local / CPU fallback

from oev.infer import OEV


def _resolve_checkpoint(spec: str) -> str:
    """Accept a local path or an 'user/repo/file.pt' Hub id."""
    if "/" not in spec or os.path.exists(spec):
        return spec
    from huggingface_hub import hf_hub_download
    repo, _, filename = spec.rpartition("/")
    return hf_hub_download(repo_id=repo, filename=filename)


CHECKPOINT = os.environ.get("OEV_CHECKPOINT", "divyanshudhruv/oev-typed/oev-base-td5.pt")
agent: OEV | None = None


def load():
    global agent
    if agent is None:
        agent = OEV(_resolve_checkpoint(CHECKPOINT), device="cuda" if torch.cuda.is_available() else "cpu")
    return agent


# ---------------- answer rendering ----------------


def _bar(label: str, prob: float) -> str:
    filled = int(round(prob * 20))
    return f"{label}  {'█' * filled}{'░' * (20 - filled)}  {prob * 100:.0f}%"


def _card(name: str, a) -> str:
    if isinstance(a, dict) and "probabilities" in a:
        conf = a.get("confidence")
        top = a.get("choice", a.get("value"))
        head = f"**{name}** → **{top}**"
        if conf is not None:
            head += f" · {conf:.0%}"
            if conf < 0.35:
                head += " · *uncertain*"
        return head + "\n\n" + "\n\n".join(_bar(k, v) for k, v in a["probabilities"].items())
    return f"**{name}** → **{a:.2f}**"


def _fmt(result) -> str:
    return "\n\n---\n\n".join(_card(n, v) for n, v in result.items())


def _ask(state: str, questions: dict) -> str:
    try:
        return _fmt(load().decide(state, questions))
    except Exception as e:
        return f"**Error**: {type(e).__name__}: {e}"


# ---------------- trained workflow presets (verbatim schemas) ----------------

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


def _customer_state(tier, tickets, msg):
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


def _invoice_state(vendor, inv, po, recv, ordered, disputes):
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


def _security_state(rule, desc, evid, crit, ptype):
    return json.dumps({
        "alert": {"description": desc, "evidence": evid, "rule": rule},
        "context": {"asset_criticality": crit, "change_window_active": False},
        "history": {"prior_incidents_90d": 0},
        "principal": {"department": "engineering", "roles": ["service"], "type": ptype},
    })


# ---------------- UI ----------------

with gr.Blocks(title="OEV") as demo:
    gr.Markdown("# OEV\n*Typed questions in. Calibrated probabilities out. One forward pass.*")

    with gr.Tab("Agent run"):
        gr.Markdown("Should an autonomous agent run continue, be observed, or be stopped?")
        st = gr.Textbox(label="trace state (JSON)", value=AGENT_STATE, lines=5)
        gr.Button("Ask", variant="primary").click(
            lambda s: _ask(s, AGENT_QS), st, gr.Textbox(label="distributions", lines=10))

    with gr.Tab("Support ticket"):
        gr.Markdown("Classify intent, score churn risk, decide if a human must step in.")
        tier = gr.Radio(["free", "business", "enterprise"], value="enterprise", label="account tier")
        tickets = gr.Slider(0, 20, value=2, step=1, label="prior tickets (90d)")
        msg = gr.Textbox(label="customer message", lines=3, value="Hello, after five years on the Enterprise plan I have decided it is time to close my account. Could you please start the cancellation process?")
        gr.Button("Ask", variant="primary").click(
            _gpu(lambda t, k, m: _ask(_customer_state(t, k, m), CUSTOMER_QS)),
            [tier, tickets, msg], gr.Textbox(label="distributions", lines=10))

    with gr.Tab("Invoice check"):
        gr.Markdown("Match invoice against purchase order and delivery; approve, hold or escalate.")
        vendor = gr.Textbox(label="vendor", value="Acme Fabrication")
        inv = gr.Number(label="invoice total (USD)", value=300020.0)
        po = gr.Number(label="purchase order total (USD)", value=300020.0)
        recv = gr.Number(label="received qty", value=1000, precision=0)
        ordered = gr.Number(label="ordered qty", value=1000, precision=0)
        disputes = gr.Slider(0, 10, value=0, step=1, label="vendor disputes (12m)")
        gr.Button("Ask", variant="primary").click(
            _gpu(lambda v, i, p, r, o, d: _ask(_invoice_state(v, i, p, r, o, d), INVOICE_QS)),
            [vendor, inv, po, recv, ordered, disputes], gr.Textbox(label="distributions", lines=10))

    with gr.Tab("Security alert"):
        gr.Markdown("Triage an alert: true positive? compromised? contain, investigate or close?")
        rule = gr.Textbox(label="detection rule", value="dormant_account_use")
        desc = gr.Textbox(label="alert description", lines=2, value="a long-unused account became active")
        evid = gr.Textbox(label="evidence", lines=3, value="The unprivileged service account `svc_task_alpha` initiated a session from IP `198.51.100.24` following 180 days of zero activity. Authentication was successful without MFA using credentials that were last rotated six months ago.")
        crit = gr.Radio(["low", "medium", "high"], value="low", label="asset criticality")
        ptype = gr.Radio(["service_account", "user"], value="service_account", label="principal type")
        gr.Button("Ask", variant="primary").click(
            _gpu(lambda r, d, e, c, p: _ask(_security_state(r, d, e, c, p), SECURITY_QS)),
            [rule, desc, evid, crit, ptype], gr.Textbox(label="distributions", lines=10))

    with gr.Tab("Playground"):
        gr.Markdown("Any state, any questions. Out-of-distribution inputs get honest, uncertain answers.")
        pg_state = gr.Textbox(label="state", lines=4, placeholder="We were charged twice for the same order.")
        pg_choice = gr.Textbox(label="choice question", placeholder="department: billing, technical, other")
        pg_noul = gr.Textbox(label="yes / no question", placeholder="refund_requested")
        pg_score = gr.Textbox(label="score question", placeholder="severity: 1-5")

        @ _gpu
        def _pg(state, cq, nq, sq):
            questions = {}
            if cq.strip():
                name, _, opts = cq.partition(":")
                questions[name.strip()] = {"type": "choice",
                                           "options": [o.strip() for o in opts.split(",")],
                                           "instructions": name.strip()}
            if nq.strip():
                questions[nq.strip()] = {"type": "noul"}
            if sq.strip():
                name, _, lv = sq.partition(":")
                lo, _, hi = lv.strip().partition("-")
                questions[name.strip()] = {"type": "score",
                                           "levels": [str(i) for i in range(int(lo), int(hi) + 1)]}
            if not state.strip() or not questions:
                return "Add a state and at least one question."
            return _ask(state, questions)

        gr.Button("Ask", variant="primary").click(_pg, [pg_state, pg_choice, pg_noul, pg_score],
                                                  gr.Textbox(label="distributions", lines=10))

    gr.Markdown("*[divyanshudhruv/oev-typed](https://huggingface.co/divyanshudhruv/oev-typed) · 184M params · one forward pass per question set*")

PORT = int(os.environ.get("OEV_PORT") or os.environ.get("PORT") or 7860) or 7860

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=PORT)
else:
    demo.launch()
