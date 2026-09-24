# OEV HF Space demo - dark, card-based interface.
# Preset workflows use trained schemas from typed-decisions; the playground
# accepts a JSON question editor like Kev's; results render as cards with
# probability bars. Nothing is generated: one forward pass per question set.

import json
import os
import time

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


# ---------------- CSS: dark theme, cards, bars ----------------

CSS = """
.gradio-container { background: #111214 !important; color: #e8e8ea !important; }
.gradio-container .prose { color: #e8e8ea !important; }
h1, h2, h3 { color: #ffffff !important; }
#header { text-align: center; padding: 18px 0 6px 0; }
#header h1 { font-size: 2.1em; margin-bottom: 2px; }
#header p { color: #9aa0a6 !important; font-size: 1.02em; margin: 0; }
#toolbar { display: flex; gap: 8px; align-items: center; justify-content: flex-end; margin: 4px 0 10px 0; }
#toolbar > div { max-width: 260px !important; }
#toolbar button.primary, #toolbar button[class*='primary'] { background: #4f9cf9 !important; border: none !important; color: #fff !important; }
button.lg.primary, .gradio-container button.primary { background: #4f9cf9 !important; border: none !important; color: #ffffff !important; }
.gradio-container button.primary:hover { background: #3f8ae8 !important; }
#statusline { color: #9aa0a6 !important; font-size: 0.88em; text-align: right; margin: -6px 0 10px 0; }
.qcard, .result-card {
    background: #1b1d21 !important;
    border: 1px solid #2a2d33 !important;
    border-radius: 12px !important;
    padding: 14px 16px !important;
    margin-bottom: 10px !important;
}
.qname { font-weight: 700; color: #ffffff; font-size: 1.02em; }
.qtype { color: #7f8691; font-size: 0.8em; text-transform: uppercase; letter-spacing: 0.06em; }
.ansname { color: #4f9cf9; font-weight: 700; font-size: 1.15em; }
.confchip {
    display: inline-block; margin-left: 8px; padding: 1px 8px; border-radius: 10px;
    font-size: 0.78em; background: #23303f; color: #8ec2ff;
}
.confchip.low { background: #3a2b23; color: #f0b477; }
.bar-row { display: flex; align-items: center; gap: 10px; margin: 5px 0; }
.bar-label { width: 190px; min-width: 190px; font-size: 0.85em; color: #cfd3d9;
             overflow: hidden; text-overflow: ellipsis; white-space: nowrap; text-align: right; }
.bar-track { flex: 1; height: 8px; background: #26292f; border-radius: 5px; overflow: hidden; }
.bar-fill { height: 100%; background: #4f9cf9; border-radius: 5px; }
.bar-fill.win { background: #57b98c; }
.bar-val { width: 48px; min-width: 48px; font-size: 0.82em; color: #9aa0a6; text-align: right; }
footer { visibility: hidden; }
"""

_TYPE_LABEL = {"choice": "choice", "noul": "yes / no", "score": "score"}


def _bar_row(label, prob, winner):
    pct = f"{prob * 100:.1f}"
    cls = "bar-fill win" if winner else "bar-fill"
    short = label if len(label) <= 26 else label[:25] + "…"
    return (f'<div class="bar-row"><div class="bar-label" title="{label}">{short}</div>'
            f'<div class="bar-track"><div class="{cls}" style="width:{prob * 100:.1f}%"></div></div>'
            f'<div class="bar-val">{pct}%</div></div>')


def _card(name, qtype, answer) -> str:
    if isinstance(answer, dict) and "probabilities" in answer:
        conf = answer.get("confidence")
        top = answer.get("choice", answer.get("value"))
        chip = ""
        if conf is not None:
            cls = "confchip low" if conf < 0.35 else "confchip"
            chip = f'<span class="{cls}">confidence {conf:.2f}</span>'
        rows = sorted(answer["probabilities"].items(), key=lambda kv: -kv[1])
        winner = rows[0][0] if rows else None
        bars = "".join(_bar_row(k, v, k == winner) for k, v in rows)
        return (f'<div class="result-card"><span class="qname">{name}</span>'
                f'<span class="qtype"> · {_TYPE_LABEL.get(qtype, qtype)}</span><br>'
                f'<span class="ansname">{top}</span>{chip}'
                f'<div style="margin-top:8px">{bars}</div></div>')
    # noul: a bare probability
    p = float(answer)
    cls = "confchip low" if 0.35 < p < 0.65 else "confchip"
    return (f'<div class="result-card"><span class="qname">{name}</span>'
            f'<span class="qtype"> · yes / no</span><br>'
            f'<span class="ansname">{"yes" if p >= 0.5 else "no"}</span>'
            f'<span class="{cls}">p(yes) {p:.2f}</span>'
            f'<div style="margin-top:8px">{_bar_row("yes", p, p >= 0.5)}{_bar_row("no", 1 - p, p < 0.5)}</div></div>')


def _render(result, n_tokens=None, latency_ms=None) -> str:
    cards = []
    for name, qtype, ans in result:
        cards.append(_card(name, qtype, ans))
    return "".join(cards)


# ---------------- inference wrapper ----------------

def _count_tokens(agent, state, questions):
    try:
        n = 0
        for name, q in questions.items():
            pq = agent._question(name, q)
            if agent.packer is not None:
                ids, _, _ = agent.packer.pack(state, pq, agent.max_len)
            else:
                from oev.dataset import pack
                ids, _, _ = pack(state, pq, agent.max_len)
            n += len(ids)
        return n
    except Exception:
        return None


def _decide(state, questions, temperature=1.0):
    a = load()
    old_t = a.temperature
    a.temperature = float(temperature)
    t0 = time.perf_counter()
    result = a.decide(state, questions)
    ms = (time.perf_counter() - t0) * 1000
    a.temperature = old_t
    qtypes = {n: q["type"] for n, q in questions.items()}
    triples = [(n, qtypes.get(n, "choice"), v) for n, v in result.items()]
    n_tok = _count_tokens(a, state, questions)
    return triples, n_tok, ms


def _error_card(msg) -> str:
    return (f'<div class="result-card"><span class="ansname" style="color:#f0776b">error</span>'
            f'<div style="margin-top:6px;color:#cfd3d9">{msg}</div></div>')


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


PLAYGROUND_STATE = "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card. What are you going to do about this?"
PLAYGROUND_QS = {
    "department": {"type": "choice",
                   "instructions": "Which team should handle this?",
                   "options": ["billing", "shipping", "other"]},
    "escalate": {"type": "noul", "instructions": "Does this need urgent human attention?"},
    "frustration": {"type": "score", "levels": ["calm", "frustrated", "very angry"]},
}

EXAMPLE_TABS = ["Support triage", "News article", "Review rating", "Return window",
                "Isolation probe", "Boundary forgery"]

# ---------------- UI ----------------

with gr.Blocks(title="OEV") as demo:
    with gr.Column(elem_id="header"):
        gr.Markdown("# OEV\n*Typed questions in. Calibrated probabilities out. One forward pass. Nothing is generated.*")

    with gr.Row(elem_id="toolbar"):
        temp = gr.Slider(0.5, 3.0, value=1.0, step=0.05, label="temperature",
                         info="<1 sharpens, >1 flattens; argmax never changes",
                         container=False, scale=1, min_width=160)
        order_btn = gr.Button("Order check", size="sm", scale=0)
        decide_btn = gr.Button("Decide", variant="primary", scale=0, min_width=140)

    status = gr.Markdown("", elem_id="statusline")

    with gr.Row():
        with gr.Column(scale=5):
            gr.Markdown("### Examples")
            with gr.Row():
                ex_support = gr.Button("Support triage", size="sm")
                ex_agent = gr.Button("Agent run", size="sm")
                ex_invoice = gr.Button("Invoice check", size="sm")
                ex_security = gr.Button("Security alert", size="sm")
                ex_play = gr.Button("Blank playground", size="sm")

            state_box = gr.Textbox(label="State  -  the document the model reads",
                                   lines=7, value=AGENT_STATE,
                                   placeholder="Plain text or a JSON object.")
            qs_box = gr.Code(label="Questions  (JSON)", language="json",
                             value=json.dumps(AGENT_QS, indent=2), lines=18)

        with gr.Column(scale=4):
            gr.Markdown("### Answers")
            results = gr.HTML('<div class="result-card" style="color:#9aa0a6">Pick an example, '
                              'edit the state or questions, then press Decide.</div>')

    # ---- verification tab stays as its own page ----
    with gr.Tab("Verify the architecture"):
        gr.Markdown(
            "Live checks on the packed-sequence design. "
            "**Isolation**: a secret in one question's instructions must not raise the probe's "
            "probability of naming it above chance. **Forgery**: anchor tokens, delimiter "
            "lookalikes and JSON injection in option text must not change how many anchors "
            "the head scores. **Order**: argmax stability under option rotation.")
        verify_out = gr.Textbox(label="results", lines=14)

        @ _gpu
        def _verify():
            import io
            import contextlib
            from oev import probes as pr

            a = load()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                pr.isolation(a.model, a.packer, a.device, repeats=2)
                print()
                pr.forgery(a.model, a.packer, a.device)
                print()
                pr.order(a.model, a.packer, a.device, rotations=6)
            return buf.getvalue()

        gr.Button("Run checks", variant="primary").click(_verify, None, verify_out)

    # ---- wiring ----

    def _parse(state, qs_json, t):
        try:
            questions = json.loads(qs_json)
        except json.JSONDecodeError as e:
            return _error_card(f"questions JSON: {e}"), ""
        if not state.strip() or not questions:
            return _error_card("Add a state and at least one question."), ""
        try:
            triples, n_tok, ms = _decide(state, questions, t)
        except Exception as e:
            return _error_card(f"{type(e).__name__}: {e}"), ""
        status = f"{len(questions)} question(s) · {n_tok if n_tok else '?'} input tokens · {ms:.0f} ms · no text generated"
        return _render(triples), status

    decide_btn.click(_gpu(_parse), [state_box, qs_box, temp], [results, status])

    def _fill(state, qs):
        return state, json.dumps(qs, indent=2)

    ex_agent.click(_fill, [gr.State(AGENT_STATE), gr.State(AGENT_QS)], [state_box, qs_box])
    ex_support.click(lambda: _fill(_customer_state("enterprise", 2,
                     "Hello, after five years on the Enterprise plan I have decided it is time "
                     "to close my account. Could you please start the cancellation process?"),
                     CUSTOMER_QS), None, [state_box, qs_box])
    ex_invoice.click(_fill, [gr.State(_invoice_state("Acme Fabrication", 300020.0, 300020.0,
                     1000, 1000, 0)), gr.State(INVOICE_QS)], [state_box, qs_box])
    ex_security.click(_fill, [gr.State(_security_state(
        "dormant_account_use", "a long-unused account became active",
        "The unprivileged service account `svc_task_alpha` initiated a session from IP "
        "`198.51.100.24` following 180 days of zero activity. Authentication was successful "
        "without MFA using credentials that were last rotated six months ago.",
        "low", "service_account")), gr.State(SECURITY_QS)], [state_box, qs_box])
    ex_play.click(_fill, [gr.State(PLAYGROUND_STATE), gr.State(PLAYGROUND_QS)],
                  [state_box, qs_box])

    @ _gpu
    def _order_check(state, qs_json, t):
        # rotate the first choice question's options and compare answers
        try:
            questions = json.loads(qs_json)
        except json.JSONDecodeError as e:
            return _error_card(f"questions JSON: {e}"), ""
        first = next((n for n, q in questions.items() if q.get("type") == "choice"), None)
        if first is None:
            return _error_card("Option-order check needs at least one choice question."), ""
        q = questions[first]
        k = len(q["options"])
        rows = []
        for r in range(min(k, 6)):
            rot = q["options"][r:] + q["options"][:r]
            rq = dict(q, options=rot)
            try:
                triples, _, _ = _decide(state, {first: rq}, t)
            except Exception as e:
                return _error_card(f"{type(e).__name__}: {e}"), ""
            ans = dict(triples[0][2])
            rows.append((r, ans.get("choice")))
        picks = [c for _, c in rows]
        stable = len(set(picks)) == 1
        html = (f'<div class="result-card"><span class="qname">order check · {first}</span>'
                f'<span class="qtype"> · {len(rows)} rotations</span><br>'
                f'<span class="ansname" style="color:{"#57b98c" if stable else "#f0b477"}">'
                f'{"stable" if stable else "flips across rotations"}</span>'
                f'<div style="margin-top:8px">'
                + "".join(f'<div class="bar-row"><div class="bar-label">rotation {r}</div>'
                          f'<div class="bar-track"></div><div class="bar-val">{c}</div></div>'
                          for r, c in rows)
                + '</div></div>')
        status = f"order check · {len(rows)} rotations of '{first}' · {'stable' if stable else 'FLIPPED'}"
        return html, status

    order_btn.click(_order_check, [state_box, qs_box, temp], [results, status])

    gr.Markdown("*[divyanshudhruv/oev-typed](https://huggingface.co/divyanshudhruv/oev-typed) · 184M params · one forward pass per question set · [probes](https://github.com/divyanshudhruv/oev/blob/main/oev/probes.py)*")

PORT = int(os.environ.get("OEV_PORT") or os.environ.get("PORT") or 7860) or 7860

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=PORT, css=CSS)
else:
    demo.launch(css=CSS)
