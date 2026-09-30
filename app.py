# OEV HF Space demo: Gradio UI with restrained styling.
#
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
    def _gpu(fn):
        return fn  # local / CPU fallback  # local / CPU fallback

from oev.infer import OEV
from space_ui import css as ui_css
from space_ui import schemas as ui_schemas
from space_ui import texts as ui_texts

# re-exported for the UI shell tests
CSS = ui_css.CSS


def _resolve_checkpoint(spec: str) -> str:
    if "/" not in spec or os.path.exists(spec):
        return spec
    from huggingface_hub import hf_hub_download

    repo, _, filename = spec.rpartition("/")
    return hf_hub_download(repo_id=repo, filename=filename)


CHECKPOINT = os.environ.get("OEV_CHECKPOINT", "divyanshudhruv/oev-typed/student-r3-oev-tiny.pt")
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


# inference plumbing

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
    except Exception:  # noqa: BLE001 - probe: any tokenizer failure means the packer can't handle the state
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
        raise TypeError("model result must be an object")
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
            raise TypeError(f"invalid result for question {name}")
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
            criteria = question.get("criteria")
            if values is None and criteria is not None:
                # Jev/TypeSafe schema: criteria as an option-name map for
                # choice, a level-label list for score.
                if question_type == "choice":
                    if not isinstance(criteria, dict) or not criteria:
                        return None, f"question {name} (choice) criteria must be a non-empty object of option names"
                    values = [str(option) for option in criteria]
                    question["options"] = values
                else:
                    if not isinstance(criteria, list) or not criteria:
                        return None, f"question {name} (score) criteria must be a list of level labels"
                    values = list(criteria)
                    question["levels"] = values
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


def _coherence(payload):
    findings = []
    for if_field, if_value, then_field, expectation in ui_schemas.COHERENCE_RULES:
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
    except Exception as e:  # noqa: BLE001 - UI boundary: show any inference failure as an error card, never crash the Space
        return "", "", "", _error_markdown(f"{type(e).__name__}: {e}")
    coh = _coherence_md(_coherence(payload))
    if coh:
        md = md + "\n\n" + coh
    return json.dumps(payload, indent=2), md, st, ""


# UI

with gr.Blocks(title="OEV") as demo:
    gr.Markdown(ui_texts.TITLE_MD)

    with gr.Tabs():
# playground
        with gr.Tab("Playground"):
            with gr.Row(elem_id="presets"):
                ex_support = gr.Button("support triage", size="sm")
                ex_invoice = gr.Button("invoice check", size="sm")
                ex_agent = gr.Button("agent run", size="sm")
                ex_security = gr.Button("security alert", size="sm")
                ex_play = gr.Button("text playground", size="sm")

            with gr.Row(equal_height=False):
# column 1: input
                with gr.Column(scale=1):
                    state_box = gr.Textbox(label="State - plain text or JSON", elem_id="state-box",
                                           lines=8, value=ui_schemas.AGENT_STATE,
                                           placeholder="Any plain text or a JSON object - "
                                                       "the model reads it as-is.")
                    qs_box = gr.Code(label="Questions (JSON)", language="json",
                                     value=json.dumps(ui_schemas.AGENT_QS, indent=2), lines=13,
                                     elem_id="qs-box")
                    with gr.Accordion("Question schema help", open=False, elem_id="qs-help"):
                        gr.Markdown(ui_texts.QS_HELP)
                    temp = gr.Slider(0.1, 3.0, value=0.7, step=0.05, label="temperature",
                                     info="<1 sharpens · >1 flattens · argmax unchanged")
                    decide_btn = gr.Button("Decide", variant="primary", size="lg",
                                           elem_id="decide-btn")

# column 2: output
                with gr.Column(scale=1):
                    error_md = gr.Markdown("", visible=False, elem_id="errorbox")
                    json_out = gr.Code(label="Raw JSON", language="json",
                                       value="// press Decide - output appears here",
                                       lines=13, interactive=False)
                    bars_md = gr.Markdown("Press **Decide** - raw JSON first, "
                                          "probability bars below.",
                                          elem_classes=["bars"])
                    status_md = gr.Markdown("", elem_id="statusline")
# api
        with gr.Tab("API"):
            gr.Markdown(ui_texts.API_DOCS)


    decide_btn.click(
        _gpu(_decide), [state_box, qs_box, temp],
        [json_out, bars_md, status_md, error_md], api_name="decide", api_visibility="public",
    ).then(
        lambda e: gr.update(visible=bool(e)), [error_md], [error_md], api_visibility="private"
    )

    def _fill(state, qs):
        return state, json.dumps(qs, indent=2)

    ex_agent.click(_fill, [gr.State(ui_schemas.AGENT_STATE), gr.State(ui_schemas.AGENT_QS)], [state_box, qs_box], api_visibility="private")
    ex_support.click(_fill, [gr.State(ui_schemas.customer_state("enterprise", 2,
                     "Hello, after five years on the Enterprise plan I have decided it is time "
                     "to close my account. Could you please start the cancellation process?")),
                     gr.State(ui_schemas.CUSTOMER_QS)], [state_box, qs_box], api_visibility="private")
    ex_invoice.click(_fill, [gr.State(ui_schemas.invoice_state_discrepant()), gr.State(ui_schemas.INVOICE_QS)], [state_box, qs_box], api_visibility="private")
    ex_security.click(_fill, [gr.State(ui_schemas.security_state(
        *ui_schemas.SUSPICIOUS_LOGIN)), gr.State(ui_schemas.SECURITY_QS)], [state_box, qs_box], api_visibility="private")
    ex_play.click(_fill, [gr.State(ui_schemas.PLAYGROUND_STATE), gr.State(ui_schemas.PLAYGROUND_QS)],
                  [state_box, qs_box], api_visibility="private")


PORT = int(os.environ.get("OEV_PORT") or os.environ.get("PORT") or 7860) or 7860

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=PORT, theme=ui_css.theme, css=ui_css.CSS)
