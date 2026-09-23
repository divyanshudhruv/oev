# OEV HF Space demo - interactive decision interface.
# Point --checkpoint at a checkpoint downloaded from the Hub at startup.

import os

import gradio as gr
import torch
import torch.hub

from oev.infer import OEV

CHECKPOINT = os.environ.get("OEV_CHECKPOINT", "checkpoints_td5/oev-tiny.pt")
agent: OEV | None = None


def load():
    global agent
    if agent is None:
        agent = OEV(CHECKPOINT, device="cuda" if torch.cuda.is_available() else "cpu")
    return agent


def decide(state: str, options: str, yesno: str, levels: str):
    """Run one forward pass over the user-typed state and questions."""
    agent = load()
    questions = {}
    if options.strip():
        for line in options.strip().splitlines():
            name, _, opts = line.partition(":")
            questions[name.strip()] = {
                "type": "choice",
                "options": [o.strip() for o in opts.split(",")],
            }
    if yesno.strip():
        for name in yesno.strip().splitlines():
            questions[name.strip()] = {"type": "noul"}
    if levels.strip():
        for line in levels.strip().splitlines():
            name, _, lv = line.partition(":")
            nums = [int(x) for x in lv.split("-")] if "-" in lv else [int(x) for x in lv.split(",")]
            questions[name.strip()] = {"type": "score", "levels": nums}

    if not state.strip() or not questions:
        return "Type a state and at least one question."

    result = agent.decide(state, questions)
    lines = []
    for name, a in result.items():
        if isinstance(a, dict) and "probabilities" in a:
            dist = "  ".join(f"{k}: {v:.3f}" for k, v in a["probabilities"].items())
            lines.append(f"{name}: {a.get('choice', a.get('value'))}  (confidence {a.get('confidence', 0):.2f})\n    {dist}")
        else:
            lines.append(f"{name}: {a:.2f}")
    return "\n\n".join(lines)


demo = gr.Interface(
    fn=decide,
    inputs=[
        gr.Textbox(label="State", lines=4, placeholder="We were charged twice for the same order."),
        gr.Textbox(label="Choice questions (name: option1, option2, ...)", lines=3,
                   placeholder="department: billing, technical, sales, other"),
        gr.Textbox(label="Yes/no questions (one per line)", lines=2,
                   placeholder="refund_requested"),
        gr.Textbox(label="Score questions (name: 1-5)", lines=2,
                   placeholder="severity: 1-5"),
    ],
    outputs=gr.Textbox(label="Calibrated distributions", lines=10),
    title="OEV - System One decision model",
    description="State + typed questions in, calibrated probabilities out, one forward pass. "
                "184M params - [model card](https://huggingface.co/divyanshudhruv/oev-typed)",
    examples=[
        ["We were charged twice for the same order.", "department: billing, technical, sales, other",
         "refund_requested", "severity: 1-5"],
    ],
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
