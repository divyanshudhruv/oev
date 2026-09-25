# OEV HF Space demo: Gradio UI with restrained styling.

# State accepts text or JSON. Outputs are probability bars and raw JSON.
# API: POST /call/decide.



import html
import json
import math
import os
import threading
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

    if "/" not in spec or os.path.exists(spec):

        return spec

    from huggingface_hub import hf_hub_download

    repo, _, filename = spec.rpartition("/")

    return hf_hub_download(repo_id=repo, filename=filename)





CHECKPOINT = os.environ.get("OEV_CHECKPOINT", "divyanshudhruv/oev-typed/student-r2b-oev-tiny.pt")

agent: OEV | None = None
_agent_lock = threading.Lock()


def load():
    global agent
    if agent is None:
        with _agent_lock:
            if agent is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                agent = OEV(_resolve_checkpoint(CHECKPOINT), device=device)
    return agent






# ----------------------------------------------------------------------

# presets - verbatim trained schemas from typed-decisions

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





def _safe_probability(value):
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(probability):
        return 0.0
    return min(1.0, max(0.0, probability))


def _probability_row(label, probability, winner=False):
    label_text = html.escape(str(label), quote=True)
    probability = _safe_probability(probability)
    percentage = probability * 100
    aria_text = html.escape(f"{label}: {percentage:.1f}%", quote=True)
    winner_class = " win" if winner else ""
    return (
        f'<div class="row{winner_class}" role="img" aria-label="{aria_text}">'
        f'<span class="name">{label_text}</span>'
        f'<span class="track"><span class="fill" '
        f'style="width:{percentage:.1f}%"></span></span>'
        f'<span class="val">{percentage:5.1f}%</span></div>\n'
    )


def _render_result(result, questions):
    if not isinstance(result, dict):
        raise ValueError("model result must be an object")
    payload = {}
    markdown = []
    for name, question in questions.items():
        if name not in result:
            raise ValueError(f"missing result for question {name}")
        value = result[name]
        question_type = question.get("type", "choice")
        question_name = html.escape(str(name), quote=True)
        markdown.append(f'<span class="qname">{question_name}</span>\n')
        if question_type == "noul":
            if isinstance(value, dict):
                probability = value.get("p_yes", value.get("value"))
            else:
                probability = value
            probability = _safe_probability(probability)
            payload[name] = {
                "type": "noul",
                "answer": "yes" if probability >= 0.5 else "no",
                "p_yes": round(probability, 4),
            }
            markdown.append(_probability_row("yes", probability, probability >= 0.5))
            markdown.append(_probability_row("no", 1.0 - probability, probability < 0.5))
            continue
        if not isinstance(value, dict) or not isinstance(value.get("probabilities"), dict):
            raise ValueError(f"invalid result for question {name}")
        probabilities = {
            str(label): _safe_probability(probability)
            for label, probability in value["probabilities"].items()
        }
        if not probabilities:
            raise ValueError(f"empty probabilities for question {name}")
        answer = value.get("choice", value.get("value"))
        if answer is None:
            answer = max(probabilities, key=lambda label: probabilities[label])
        confidence = value.get("confidence")
        if confidence is None:
            confidence = max(probabilities.values())
        payload[name] = {
            "type": question_type,
            "answer": answer,
            "confidence": _safe_probability(confidence),
            "probabilities": probabilities,
        }
        rows = sorted(probabilities.items(), key=lambda item: (-item[1], str(item[0])))
        for index, (label, probability) in enumerate(rows):
            markdown.append(_probability_row(label, probability, index == 0))
    return payload, "\n".join(markdown)


def _run(state, questions, temperature=1.0):
    a = load()
    t0 = time.perf_counter()
    result = a.decide(state, questions, temperature=float(temperature))
    ms = (time.perf_counter() - t0) * 1000
    payload, markdown = _render_result(result, questions)
    n_tok = _count_tokens(a, state, questions)
    token_text = str(n_tok) if n_tok is not None else "?"
    status = (
        f"{len(questions)} question(s) · {token_text} input tokens · "
        f"{ms:.0f} ms · one forward pass per question · nothing generated"
    )
    return payload, markdown, status





def _error_markdown(error):
    message = html.escape(str(error), quote=True)
    return f"**error** - {message}"


def _valid_score_level(value):
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        try:
            return math.isfinite(value)
        except (OverflowError, TypeError):
            return False
    return isinstance(value, str) and bool(value.strip())


def _validate(state, qs_json):

    if not isinstance(state, str) or not state.strip():

        return None, "state must be non-empty text or JSON"

    try:

        questions = json.loads(qs_json)

    except (TypeError, json.JSONDecodeError) as e:

        return None, f"questions JSON: {e}"

    if not isinstance(questions, dict) or not questions:

        return None, "add at least one question"

    for name, question in questions.items():

        if not isinstance(name, str) or not name.strip():

            return None, "question names must be non-empty strings"

        if not isinstance(question, dict):

            return None, f"question {name} must be an object"

        question_type = question.get("type")

        if question_type not in {"choice", "noul", "score"}:

            return None, f"question {name} has unsupported type"

        if question_type in {"choice", "score"}:

            key = "options" if question_type == "choice" else "levels"
            values = question.get(key)

            if not isinstance(values, list) or not values or not all(
                _valid_score_level(value) if question_type == "score"
                else isinstance(value, str) and value.strip()
                for value in values
            ):

                return None, f"question {name} needs a non-empty {key} list"

            normalized_values = [str(value).strip() for value in values]

            if len(set(normalized_values)) != len(normalized_values):

                return None, f"question {name} must have unique {key}"

        if "instructions" in question and not isinstance(question["instructions"], str):

            return None, f"question {name} instructions must be text"

    return questions, None





COHERENCE_RULES = [
    ("action", "human_review", "needs_review", "yes"),
    ("action", "stop", "needs_review", "yes"),
    ("action", "continue", "needs_review", "no"),
]


def _coherence(payload):
    findings = []
    for if_field, if_value, then_field, expectation in COHERENCE_RULES:
        source = payload.get(if_field)
        if not isinstance(source, dict) or source.get("answer") != if_value:
            continue
        target = payload.get(then_field)
        if not isinstance(target, dict):
            continue
        if target.get("type") == "noul":
            p_yes = _safe_probability(target.get("p_yes"))
            ok = (p_yes >= 0.5) if expectation == "yes" else p_yes < 0.5
            description = f"{if_field} = {if_value} · {then_field} p(yes) = {p_yes:.2f}"
        else:
            answer = target.get("answer")
            ok = answer == expectation
            description = f"{if_field} = {if_value} · {then_field} = {answer}"
        findings.append((description, ok))
    return findings


def _coherence_md(findings):
    if not findings:
        return ""
    lines = ['<span class="qname">coherence</span>\n']
    for description, ok in findings:
        safe_description = html.escape(description, quote=True)
        status = "consistent" if ok else "**CONTRADICTION**"
        lines.append(f"{'pass' if ok else 'warning'}: {safe_description} - {status}\n")
    return "\n".join(lines)


def _decide(state, qs_json, t):

    questions, err = _validate(state, qs_json)

    if err:

        return "", "", "", _error_markdown(err)

    try:

        payload, md, st = _run(state, questions, t)

    except Exception as e:

        return "", "", "", _error_markdown(f"{type(e).__name__}: {e}")

    coh = _coherence_md(_coherence(payload))

    if coh:

        md = md + "\n\n" + coh

    return json.dumps(payload, indent=2), md, st, ""


def _order_check(state, qs_json, t):

    questions, err = _validate(state, qs_json)

    if err or questions is None:

        return "", "", _error_markdown(err or "questions are missing")

    first = next((n for n, q in questions.items() if q.get("type") == "choice"), None)

    if first is None:

        return "", "", _error_markdown("order check needs at least one choice question")

    q = questions[first]

    k = len(q["options"])

    rows, picks = [], []

    for r in range(min(k, 6)):

        rot = q["options"][r:] + q["options"][:r]

        rq = dict(q, options=rot)

        try:

            payload, _, _ = _run(state, {first: rq}, t)

        except Exception as e:

            return "", "", _error_markdown(f"{type(e).__name__}: {e}")

        pick = payload[first]["answer"]

        picks.append(pick)

        rows.append(f"rotation {r}: `{html.escape(str(pick), quote=True)}`")

    stable = len(set(picks)) == 1

    verdict = "**stable under rotation**" if stable else "**flips across rotations**"

    safe_first = html.escape(str(first), quote=True)

    md = '<span class="qname">order check</span>' + safe_first + chr(10).join([""] + rows) + chr(10) + chr(10) + verdict

    return "", md, f"{len(rows)} rotations of '{html.escape(str(first), quote=True)}'"






# ----------------------------------------------------------------------

# theme + a thin CSS layer (restraint: spacing, mono digits, focus rings)

# ----------------------------------------------------------------------



theme = gr.themes.Soft(

    primary_hue="orange",

    neutral_hue="slate",

    font=[gr.themes.GoogleFont("IBM Plex Sans"), "ui-sans-serif", "system-ui", "sans-serif"],

    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],

).set(

    body_background_fill="#171412",

    body_background_fill_dark="#171412",

    body_text_color="#f5eee8",

    body_text_color_dark="#f5eee8",

    body_text_color_subdued="#b9aa9f",

    body_text_color_subdued_dark="#b9aa9f",

    block_background_fill="#211d1a",

    block_background_fill_dark="#211d1a",

    block_border_color="#3a2f28",

    block_border_color_dark="#3a2f28",

    border_color_primary="#3a2f28",

    border_color_primary_dark="#3a2f28",

    color_accent="#e8b48c",

    color_accent_soft="#5a4032",

    color_accent_soft_dark="#5a4032",

    border_color_accent="#e8b48c",

    border_color_accent_dark="#e8b48c",

    input_background_fill="#1b1816",

    input_background_fill_dark="#1b1816",

    input_border_color="#4a3b32",

    input_border_color_dark="#4a3b32",

    input_border_color_focus="#e8b48c",

    input_border_color_focus_dark="#e8b48c",

    button_primary_background_fill="#e8b48c",

    button_primary_background_fill_dark="#e8b48c",

    button_primary_background_fill_hover="#dca47a",

    button_primary_background_fill_hover_dark="#dca47a",

    button_primary_border_color="#e8b48c",

    button_primary_border_color_dark="#e8b48c",

    button_primary_border_color_hover="#dca47a",

    button_primary_border_color_hover_dark="#dca47a",

    button_primary_text_color="#2b1d16",

    button_primary_text_color_dark="#2b1d16",

    button_primary_text_color_hover="#2b1d16",

    button_primary_text_color_hover_dark="#2b1d16",

    button_secondary_background_fill="#2a2420",

    button_secondary_background_fill_dark="#2a2420",

    button_secondary_background_fill_hover="#352c26",

    button_secondary_background_fill_hover_dark="#352c26",

    button_secondary_border_color="#4a3b32",

    button_secondary_border_color_dark="#4a3b32",

    button_secondary_border_color_hover="#5a4639",

    button_secondary_border_color_hover_dark="#5a4639",

    button_secondary_text_color="#f5eee8",

    button_secondary_text_color_dark="#f5eee8",

    button_secondary_text_color_hover="#ffffff",

    button_secondary_text_color_hover_dark="#ffffff",

    block_border_width="1px",

    block_radius="0px",

    button_large_radius="0px",

    button_small_radius="0px",

    input_radius="0px",

)



CSS = """

/* --- page frame: centered column --- */

.gradio-container { width: 1020px !important; max-width: calc(100vw - 32px) !important; min-width: 0 !important; box-sizing: border-box !important; margin: 0 auto; }

footer { visibility: hidden; }

:root, .gradio-container { --primary-pastel: #e8b48c; --color-accent: var(--primary-pastel); --border-color-accent: var(--primary-pastel); }

.gradio-container > main,
.gradio-container .main,
.tab-container,
.tab-container > div,
.tab-container .tab-panel,
.tab-container .row { width: 100% !important; min-width: 0 !important; box-sizing: border-box !important; }
.tab-container .column { min-width: 0 !important; box-sizing: border-box !important; }

.gradio-container { font-size: 14px; line-height: 1.5; }
.gradio-container p, .gradio-container textarea, .gradio-container input { font-size: 14px; line-height: 1.5; }
.gradio-container h1 { font-size: 24px; line-height: 1.25; font-weight: 600; }
.gradio-container h2 { font-size: 18px; line-height: 1.25; font-weight: 600; }
.gradio-container h3 { font-size: 16px; line-height: 1.25; font-weight: 600; }
.gradio-container label, .gradio-container .form-label, .gradio-container .block-label { font-size: 12px; line-height: 1.4; font-weight: 500; }
.gradio-container button, .tab-nav button { font-size: 13px; line-height: 1.25; }



/* --- everything square: nuclear override over any gradio rounding --- */

.gradio-container * { border-radius: 0 !important; }



/* --- header: compact, with a hairline rule under it --- */

#header { padding: 12px 0 16px; border-bottom: 1px solid var(--border-color-primary);

          margin-bottom: 16px; }

#header h1 { margin: 0 0 8px; letter-spacing: -0.01em; }

#header .prose p { color: var(--body-text-color-subdued); margin: 0 0 4px; }

#header .metrics { font-family: var(--font-mono); font-size: 12px;

                   color: var(--body-text-color-subdued);

                   letter-spacing: 0.02em; }

#header .metrics b { color: var(--body-text-color); font-weight: 600; }



/* --- preset buttons: equal width, one row, one height --- */

#presets { gap: 8px; margin-bottom: 16px; }

#presets button { flex: 1 1 0; min-width: 0; width: auto; height: 32px !important;

                  min-height: 32px !important; padding: 0 12px !important; }



/* --- accordion header matches the preset button height --- */


#qs-help { margin: 12px 0; }

#qs-help button { min-height: 32px !important; padding: 4px 12px !important; }



/* --- decide: full width, on its own row --- */

#decide-btn { width: 100%; min-height: 40px; margin-top: 16px; }

.tab-container .row { gap: 16px; }



/* --- code editors and viewers render mono, like code should --- */

.cm-editor, .cm-content, .cm-line, textarea.code,
#state-box textarea, #qs-box textarea {

  font-family: var(--font-mono) !important;

  font-size: 12px !important;

}

/* --- real CSS progress bars rendered from the payload markdown --- */

.bars .qname { display: block; font-size: 12px; text-transform: uppercase;

               letter-spacing: 0.1em; color: var(--body-text-color-subdued);

               margin: 16px 0 8px; }

.bars .row { display: flex; align-items: center; gap: 8px; margin: 4px 0; }

.bars .row .name { flex: 0 0 38%; overflow: hidden; text-overflow: ellipsis;

                   white-space: nowrap; font-size: 12px; }

.bars .row .track { flex: 1 1 auto; height: 8px;

                    background: var(--block-background-fill);

                    border: 1px solid var(--border-color-primary); position: relative; }

.bars .row .fill { position: absolute; inset: 0 auto 0 0; height: 100%;

                   background: var(--color-accent); opacity: 0.75; }

.bars .row.win .fill { background: var(--color-accent); opacity: 1; }

.bars .row .val { flex: 0 0 52px; text-align: right;

                  font-family: var(--font-mono); font-size: 12px;

                  color: var(--body-text-color-subdued); }

.bars .row.win .val { color: var(--body-text-color); font-weight: 600; }



/* --- status line: quiet mono --- */

#statusline { font-family: var(--font-mono); font-size: 12px;

              color: var(--body-text-color-subdued); text-align: right;

              min-height: 1.4em; margin-top: 8px; }



/* --- error: red left rule --- */

#errorbox { margin: 12px 0; padding: 12px;

            border-left: 3px solid var(--color-danger) !important; }



/* --- inputs get focus rings from the theme; deepen them slightly --- */

#state-box textarea:focus, #qs-box textarea:focus, #qs-box input:focus,
.cm-editor.cm-focused, .cm-content:focus {

  outline: 2px solid var(--border-color-accent) !important;

  outline-offset: -1px;

}



/* --- tab nav: slightly denser, mono labels --- */

.tab-nav { display: flex; width: 100%; gap: 8px; margin-bottom: 16px; }
.tab-nav button { flex: 1 1 0; min-width: 0; min-height: 32px; padding: 6px 12px; font-size: 13px; letter-spacing: 0.04em; }
.tab-nav button:hover,
.tab-nav button:focus-visible,
.tab-nav button[aria-selected="true"] {
  color: var(--color-accent) !important;
  -webkit-text-fill-color: var(--color-accent) !important;
  background: var(--color-accent-soft) !important;
  border-color: var(--color-accent) !important;
  box-shadow: inset 0 -2px 0 var(--color-accent) !important;
}
.tab-nav button:hover *,
.tab-nav button:focus-visible *,
.tab-nav button[aria-selected="true"] * {
  color: var(--color-accent) !important;
  -webkit-text-fill-color: var(--color-accent) !important;
}



/* --- responsive: presets wrap nicely on narrow screens --- */

@media (max-width: 640px) {

  #presets { flex-wrap: wrap; }

  #presets button { flex: 1 1 45%; width: auto; }

}

"""



TITLE_MD = """

<div id="header">

<h1>OEV</h1>

<p><b>Typed questions in. Calibrated probabilities out. One forward pass. Nothing is generated.</b></p>

<p class="metrics"><b>184M</b> params · calibrated distributions · <b>0</b> tokens generated</p>

</div>

"""



QS_HELP = """**Question schema** - one JSON object per question name:



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



API_DOCS = r"""## Use OEV from code



The Space exposes one public API event, `decide`. Calls use two steps: submit
inputs, then fetch the streamed result by event ID.



```bash

SPACE_URL="https://divyanshudhruv-oev-demo.hf.space"

EVENT_ID=$(curl -sS -X POST "$SPACE_URL/call/decide" \

  -H "Content-Type: application/json" \

  -d '{"data": ["state text", "{\"action\": {\"type\": \"choice\", \"options\": [\"continue\", \"stop\"]}}", 1.0]}' |

  python -c "import json,sys; print(json.load(sys.stdin)['event_id'])")

curl -N "$SPACE_URL/call/decide/$EVENT_ID"

```



The response is an output list. Its first item is a JSON object with one
entry per question. Choice and score questions include `answer`, `confidence`,
and the full `probabilities` distribution. Yes/no questions include `p_yes`



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

result = agent.decide(state, questions)

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

                with gr.Column(scale=1):

                    state_box = gr.Textbox(label="State - plain text or JSON", elem_id="state-box",

                                           lines=8, value=AGENT_STATE,

                                           placeholder="Any plain text or a JSON object - "

                                                       "the model reads it as-is.")

                    qs_box = gr.Code(label="Questions (JSON)", language="json",

                                     value=json.dumps(AGENT_QS, indent=2), lines=13,

                                     elem_id="qs-box")

                    with gr.Accordion("Question schema help", open=False, elem_id="qs-help"):

                        gr.Markdown(QS_HELP)

                    temp = gr.Slider(0.1, 3.0, value=1.0, step=0.05, label="temperature",

                                     info="<1 sharpens · >1 flattens · argmax unchanged")

                    decide_btn = gr.Button("Decide", variant="primary", size="lg",

                                           elem_id="decide-btn")



                # -------- column 2: output --------

                with gr.Column(scale=1):

                    error_md = gr.Markdown("", visible=False, elem_id="errorbox")

                    json_out = gr.Code(label="Raw JSON", language="json",

                                       value="// press Decide - output appears here",

                                       lines=13, interactive=False)

                    bars_md = gr.Markdown("Press **Decide** - raw JSON first, "

                                          "probability bars below.",

                                          elem_classes=["bars"])

                    status_md = gr.Markdown("", elem_id="statusline")

        # ================= verify =================

        with gr.Tab("Verify the architecture"):

            gr.Markdown(

                "Live checks on the packed-sequence design.\n\n"

                "**Isolation** - a secret in one question's instructions must not raise the "

                "probe's probability of naming it above chance.\n\n"

                "**Forgery** - anchor tokens, delimiter lookalikes and JSON injection in "

                "option text must not change how many anchors the head scores.\n\n"

                "**Order** - argmax stability under option rotation.")

            verify_btn = gr.Button("Run checks", variant="primary")

            verify_out = gr.Textbox(label="results", lines=14)



        # ================= order check =================

        with gr.Tab("Order check"):

            gr.Markdown("Rotates the first choice question's options and reports whether the "

                        "argmax answer moves. Uses the state and questions from the Playground "

                        "tab - edit them there first.")

            order_btn = gr.Button("Run order check", variant="primary")

            order_out = gr.Markdown(elem_classes=["bars"])



        # ================= api =================

        with gr.Tab("API"):

            gr.Markdown(API_DOCS)



    # ---- wiring ----



    decide_btn.click(

        _gpu(_decide), [state_box, qs_box, temp],

        [json_out, bars_md, status_md, error_md], api_name="decide", api_visibility="public",

    ).then(
        lambda e: gr.update(visible=bool(e)), [error_md], [error_md], api_visibility="private"
    )



    def _fill(state, qs):

        return state, json.dumps(qs, indent=2)



    ex_agent.click(_fill, [gr.State(AGENT_STATE), gr.State(AGENT_QS)], [state_box, qs_box], api_visibility="private")

    ex_support.click(_fill, [gr.State(_customer_state("enterprise", 2,

                     "Hello, after five years on the Enterprise plan I have decided it is time "

                     "to close my account. Could you please start the cancellation process?")),

                     gr.State(CUSTOMER_QS)], [state_box, qs_box], api_visibility="private")

    ex_invoice.click(_fill, [gr.State(_invoice_state("Acme Fabrication", 300020.0, 300020.0,

                     1000, 1000, 0)), gr.State(INVOICE_QS)], [state_box, qs_box], api_visibility="private")

    ex_security.click(_fill, [gr.State(_security_state(

        "dormant_account_use", "a long-unused account became active",

        "The unprivileged service account `svc_task_alpha` initiated a session from IP "

        "`198.51.100.24` following 180 days of zero activity. Authentication was successful "

        "without MFA using credentials that were last rotated six months ago.",

        "low", "service_account")), gr.State(SECURITY_QS)], [state_box, qs_box], api_visibility="private")

    ex_play.click(_fill, [gr.State(PLAYGROUND_STATE), gr.State(PLAYGROUND_QS)],

                  [state_box, qs_box], api_visibility="private")



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



    verify_btn.click(_verify, None, verify_out, api_visibility="private")



    order_btn.click(

        _gpu(_order_check), [state_box, qs_box, temp],

        [json_out, order_out, status_md], api_visibility="private"

    )





PORT = int(os.environ.get("OEV_PORT") or os.environ.get("PORT") or 7860) or 7860



if __name__ == "__main__":

    demo.launch(server_name="0.0.0.0", server_port=PORT, theme=theme, css=CSS)
