# OEV HF Space demo — v3.1: stock Gradio + a thin, restrained CSS layer.

# gr.themes.Soft() dark, gr.Tabs layout. State input has a segmented control

# (JSON / Text). Output: Markdown bars + raw JSON. API: POST /call/decide.



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





CHECKPOINT = os.environ.get("OEV_CHECKPOINT", "divyanshudhruv/oev-typed/student-r2b-oev-tiny.pt")

agent: OEV | None = None





def load():

    global agent

    if agent is None:

        agent = OEV(_resolve_checkpoint(CHECKPOINT), device="cuda" if torch.cuda.is_available() else "cpu")

    return agent





# ----------------------------------------------------------------------

# presets — verbatim trained schemas from typed-decisions

# ----------------------------------------------------------------------



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



AGENT_TEXT = None  # text presets folded into the single state box (JSON accepted inline)



# ----------------------------------------------------------------------

# inference plumbing

# ----------------------------------------------------------------------



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





def _run(state, questions, temperature=1.0):

    """Shared inference. Returns (payload_dict, markdown_bars, status_line)."""

    a = load()

    old_t = a.temperature

    a.temperature = float(temperature)

    t0 = time.perf_counter()

    result = a.decide(state, questions)

    ms = (time.perf_counter() - t0) * 1000

    a.temperature = old_t



    payload, md = {}, []

    for name, q in questions.items():

        v = result.get(name)

        qtype = q.get("type", "choice")

        if isinstance(v, dict) and "probabilities" in v:

            payload[name] = {"type": qtype,

                             "answer": v.get("choice", v.get("value")),

                             "confidence": v.get("confidence"),

                             "probabilities": v["probabilities"]}

            md.append(f"### {name}\n")

            rows = sorted(v["probabilities"].items(), key=lambda kv: -kv[1])

            for opt, p in rows:

                blocks = max(1, round(p * 20))

                md.append(f"`{'█' * blocks:<20}` {p * 100:5.1f}%  — {opt}\n")

        else:

            p = float(v)

            payload[name] = {"type": "noul",

                             "answer": "yes" if p >= 0.5 else "no",

                             "p_yes": round(p, 4)}

            md.append(f"### {name}\n")

            blocks = max(1, round(p * 20))

            md.append(f"`{'█' * blocks:<20}` {p * 100:5.1f}%  — yes\n")

            md.append(f"`{'░' * (20 - blocks):<20}` {100 - p * 100:5.1f}%  — no\n")

    n_tok = _count_tokens(a, state, questions)

    status = (f"{len(questions)} question(s) · {n_tok if n_tok else '?'} input tokens · "

              f"{ms:.0f} ms · one forward pass · nothing generated")

    return payload, "\n".join(md), status





def _validate(state, qs_json):

    try:

        questions = json.loads(qs_json)

    except json.JSONDecodeError as e:

        return None, f"questions JSON: {e}"

    if not isinstance(questions, dict) or not questions:

        return None, "add at least one question"

    return questions, None





# ---- coherence check (policy layer per COHERENCE_TAB.md) ----
# Declarative rules: if <field> answers <value>, then <other field> should agree.
# Findings render under the probability bars; contradictions are flagged.

COHERENCE_RULES = [
    # (if_field, if_value, then_field, expectation)  expectation: "yes"/"no" for noul p>=0.5, or exact option text
    ("action", "human_review", "needs_review", "yes"),
    ("action", "stop", "needs_review", "yes"),
    ("action", "continue", "needs_review", "no"),
]


def _coherence(payload):
    """Return [(description, ok_bool)] findings for one decide() payload."""
    findings = []
    for if_field, if_val, then_field, expect in COHERENCE_RULES:
        a = payload.get(if_field, {}).get("answer")
        if a != if_val or then_field not in payload:
            continue
        b = payload[then_field]
        if b.get("type") == "noul":
            ok = (b.get("p_yes", 0) >= 0.5) if expect == "yes" else (b.get("p_yes", 1) < 0.5)
            desc = f"{if_field} = {if_val} · {then_field} p(yes) = {b.get('p_yes', 0):.2f}"
        else:
            ok = b.get("answer") == expect
            desc = f"{if_field} = {if_val} · {then_field} = {b.get('answer')}"
        findings.append((desc, ok))
    return findings


def _coherence_md(findings):
    if not findings:
        return ""
    lines = ["### coherence\n"]
    for desc, ok in findings:
        lines.append(f"{'✅' if ok else '⚠️'} {desc} — {'consistent' if ok else '**CONTRADICTION**'}\n")
    return "\n".join(lines)


def _decide(state, qs_json, t):

    """Main handler: returns (payload_json_str, markdown, status, error_markdown)."""

    questions, err = _validate(state, qs_json)

    if err:

        return "", "", "", f"**error** — {err}"

    try:

        payload, md, st = _run(state, questions, t)

    except Exception as e:

        return "", "", "", f"**error** — {type(e).__name__}: {e}"

    coh = _coherence_md(_coherence(payload))

    if coh:

        md = md + "\n\n" + coh

    return json.dumps(payload, indent=2), md, st, ""





def _order_check(state, qs_json, t):

    """Rotate the first choice question's options; report argmax stability."""

    questions, err = _validate(state, qs_json)

    if err:

        return "", "", f"**error** — {err}"

    first = next((n for n, q in questions.items() if q.get("type") == "choice"), None)

    if first is None:

        return "", "", "**error** — order check needs at least one choice question"

    q = questions[first]

    k = len(q["options"])

    rows, picks = [], []

    for r in range(min(k, 6)):

        rot = q["options"][r:] + q["options"][:r]

        rq = dict(q, options=rot)

        try:

            payload, _, _ = _run(state, {first: rq}, t)

        except Exception as e:

            return "", "", f"**error** — {type(e).__name__}: {e}"

        pick = payload[first]["answer"]

        picks.append(pick)

        rows.append(f"rotation {r}: `{pick}`")

    stable = len(set(picks)) == 1

    verdict = "✅ **stable under rotation**" if stable else "⚠️ **flips across rotations**"

    md = "### order check - " + first + chr(10).join([""] + rows) + chr(10) + chr(10) + verdict

    return "", md, f"{len(rows)} rotations of '{first}'"





# ----------------------------------------------------------------------

# theme + a thin CSS layer (restraint: spacing, mono digits, focus rings)

# ----------------------------------------------------------------------



theme = gr.themes.Soft(

    primary_hue="neutral",

    neutral_hue="slate",

    font=[gr.themes.GoogleFont("IBM Plex Sans"), "ui-sans-serif", "system-ui", "sans-serif"],

    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],

).set(

    body_background_fill_dark="#0e0e11",

    block_background_fill_dark="#16161b",

    block_border_width="1px",

    block_radius="6px",

    button_large_radius="6px",

    button_small_radius="6px",

    input_radius="6px",

)



CSS = """

/* --- page frame: centered column, quiet gradient backdrop --- */

.gradio-container { max-width: 920px !important; margin: 0 auto; }

footer { visibility: hidden; }



/* --- header: compact, with a hairline rule under it --- */

#header { padding: 6px 0 14px; border-bottom: 1px solid var(--border-color-primary);

          margin-bottom: 6px; }

#header h1 { margin: 0 0 6px; letter-spacing: -0.01em; }

#header .prose p { color: var(--body-text-color-subdued); margin: 2px 0; }

#header .metrics { font-family: var(--font-mono); font-size: 0.82em;

                   color: var(--body-text-color-subdued);

                   letter-spacing: 0.02em; }

#header .metrics b { color: var(--body-text-color); font-weight: 600; }



/* --- preset buttons: equal width, one row --- */

#presets { gap: 8px; }

#presets button { min-width: 0; width: 100%; }



/* --- probability bars: mono, aligned, readable --- */

.bars code {

  font-family: var(--font-mono) !important;

  font-size: 0.86em !important;

  letter-spacing: 0.04em;

  background: transparent !important;

  padding: 0 !important;

}

.bars h3 { font-size: 0.92em !important; text-transform: uppercase;

           letter-spacing: 0.1em; color: var(--body-text-color-subdued) !important;

           margin: 18px 0 6px !important; }

.bars .win { color: var(--body-text-color); font-weight: 700; }



/* --- status line: quiet mono --- */

#statusline { font-family: var(--font-mono); font-size: 0.78em;

              color: var(--body-text-color-subdued); text-align: right;

              min-height: 1.4em; }



/* --- error: red left rule --- */

#errorbox { border-left: 3px solid var(--color-danger) !important;

            padding-left: 12px; }



/* --- inputs get focus rings from the theme; deepen them slightly --- */

#state-json textarea:focus, #state-text textarea:focus,

#qs-box textarea:focus, #qs-box input:focus {

  outline: 2px solid var(--border-color-accent) !important;

  outline-offset: -1px;

}



/* --- tab nav: slightly denser, mono labels --- */

.tab-nav button { font-size: 0.86em !important; letter-spacing: 0.04em; }



/* --- responsive: presets wrap nicely on narrow screens --- */

@media (max-width: 640px) {

  #presets { flex-wrap: wrap; }

  #presets button { min-width: 45%; }

}

"""



TITLE_MD = """

<div id="header">

<h1>OEV</h1>

<p><b>Typed questions in. Calibrated probabilities out. One forward pass. Nothing is generated.</b></p>

<p class="metrics"><b>184M</b> params · <b>22.2 ms</b> per decision on a T4 · <b>0</b> tokens generated</p>

</div>

"""



QS_HELP = """**Question schema** — one JSON object per question name:



```json

{

  "question_name": {

    "type": "choice",          // "choice" | "noul" (yes/no) | "score" (ordinal)

    "options": ["a", "b"],     // choice: the labels; score: level descriptions

    "levels": ["low", "high"], // score only: ordinal level descriptions

    "instructions": "what to decide"

  }

}

```

"""



API_DOCS = """## Use OEV from code



The Space exposes a REST endpoint — no browser needed:



```bash

curl -X POST https://divyanshudhruv-oev-demo.hf.space/call/decide \\

  -H "Content-Type: application/json" \\

  -d '{

    "data": [

      "{\\"task\\": \\"summarize the weekly sales dashboard\\"}",

      "{\\"action\\": {\\"type\\": \\"choice\\", \\"options\\": [\\"continue\\", \\"stop\\"]}}",

      1.0

    ]

  }'

```



Response: a JSON object, one entry per question — `answer`, `confidence`, and the

full `probabilities` distribution (or `p_yes` for yes/no questions).



Or with the `gradio_client`:



```python

from gradio_client import Client

client = Client("divyanshudhruv/oev-demo")

result = client.predict(

    state,          # str: plain text or JSON document

    questions,      # str: JSON, {"name": {"type": "choice|noul|score", ...}}

    1.0,            # float: temperature

    api_name="/decide",

)

```



Python (local install):



```python

from oev.infer import OEV

agent = OEV("divyanshudhruv/oev-typed/student-r2b-oev-tiny.pt")

result = agent.decide(state, questions)   # one forward pass, full distributions

```

"""



# ----------------------------------------------------------------------

# UI

# ----------------------------------------------------------------------



with gr.Blocks(title="OEV") as demo:

    gr.Markdown(TITLE_MD)



    with gr.Tabs():

        # ================= playground =================

        with gr.Tab("Playground"):

            with gr.Row(elem_id="presets"):

                ex_agent = gr.Button("agent run", size="sm")

                ex_support = gr.Button("support triage", size="sm")

                ex_invoice = gr.Button("invoice check", size="sm")

                ex_security = gr.Button("security alert", size="sm")

                ex_play = gr.Button("text playground", size="sm")



            with gr.Row(equal_height=False):

                # -------- column 1: input --------

                with gr.Column(scale=5):

                    state_box = gr.Textbox(label="State — plain text or JSON",

                                           lines=8, value=AGENT_STATE,

                                           placeholder="Any plain text or a JSON object — "

                                                       "the model reads it as-is.")

                    qs_box = gr.Code(label="Questions (JSON)", language="json",

                                     value=json.dumps(AGENT_QS, indent=2), lines=13,

                                     elem_id="qs-box")

                    with gr.Accordion("Question schema help", open=False):

                        gr.Markdown(QS_HELP)

                    with gr.Row():

                        temp = gr.Slider(0.5, 3.0, value=1.0, step=0.05, label="temperature",

                                         info="<1 sharpens · >1 flattens · argmax unchanged")

                        decide_btn = gr.Button("Decide", variant="primary", size="lg", scale=0)



                # -------- column 2: output --------

                with gr.Column(scale=4):

                    error_md = gr.Markdown("", visible=False, elem_id="errorbox")

                    bars_md = gr.Markdown("Press **Decide** — one section per question, "

                                          "probability bars in plain Markdown.",

                                          elem_classes=["bars"])

                    json_out = gr.Code(label="Raw JSON", language="json", lines=13,

                                       interactive=False)

                    status_md = gr.Markdown("", elem_id="statusline")



        # ================= verify =================

        with gr.Tab("Verify the architecture"):

            gr.Markdown(

                "Live checks on the packed-sequence design.\n\n"

                "**Isolation** — a secret in one question's instructions must not raise the "

                "probe's probability of naming it above chance.\n\n"

                "**Forgery** — anchor tokens, delimiter lookalikes and JSON injection in "

                "option text must not change how many anchors the head scores.\n\n"

                "**Order** — argmax stability under option rotation.")

            verify_btn = gr.Button("Run checks", variant="primary")

            verify_out = gr.Textbox(label="results", lines=14)



        # ================= order check =================

        with gr.Tab("Order check"):

            gr.Markdown("Rotates the first choice question's options and reports whether the "

                        "argmax answer moves. Uses the state and questions from the Playground "

                        "tab — edit them there first.")

            order_btn = gr.Button("Run order check", variant="primary")

            order_out = gr.Markdown(elem_classes=["bars"])



        # ================= api =================

        with gr.Tab("API"):

            gr.Markdown(API_DOCS)



    # ---- wiring ----



    decide_btn.click(

        _gpu(_decide), [state_box, qs_box, temp],

        [json_out, bars_md, status_md, error_md], api_name="decide",

    ).then(lambda e: gr.update(visible=bool(e)), [error_md], [error_md])



    def _fill(state, qs):

        return state, json.dumps(qs, indent=2)



    ex_agent.click(_fill, [gr.State(AGENT_STATE), gr.State(AGENT_QS)], [state_box, qs_box])

    ex_support.click(_fill, [gr.State(_customer_state("enterprise", 2,

                     "Hello, after five years on the Enterprise plan I have decided it is time "

                     "to close my account. Could you please start the cancellation process?")),

                     gr.State(CUSTOMER_QS)], [state_box, qs_box])

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



    verify_btn.click(_verify, None, verify_out)



    order_btn.click(

        _gpu(_order_check), [state_box, qs_box, temp],

        [json_out, order_out, status_md],

    )





PORT = int(os.environ.get("OEV_PORT") or os.environ.get("PORT") or 7860) or 7860



if __name__ == "__main__":

    demo.launch(server_name="0.0.0.0", server_port=PORT, theme=theme, css=CSS)

else:

    demo.launch(theme=theme, css=CSS)