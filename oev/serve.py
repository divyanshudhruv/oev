"""Minimal HTTP server for OEV decisions.

Usage:
    pip install fastapi uvicorn
    python -m oev.serve --checkpoint checkpoints_td5/oev-tiny.pt --port 8000

    # native schema:
    curl -X POST localhost:8000/decide -H "Content-Type: application/json" -d '{
      "state": "We were charged twice for the same order.",
      "questions": {
        "department": {"type": "choice", "options": ["billing", "technical", "other"],
                       "instructions": "Which department?"},
        "refund_requested": {"type": "noul"}
      }
    }'

    # Jev-compatible schema (drop-in for existing TypeSafe clients):
    curl -X POST localhost:8000/v1/systemone -H "Content-Type: application/json" -d '{
      "model": "oev",
      "state": "We were charged twice for the same order.",
      "questions": {
        "refund_requested": {"type": "noul", "instructions": "Does the user request a refund?"}
      }
    }'
"""

import argparse

from fastapi import FastAPI
from pydantic import BaseModel, Field

from oev.infer import OEV
from oev.presets import gate

app = FastAPI(title="OEV", description="System One decision model: typed questions in, calibrated probabilities out.")
agent: OEV | None = None


class DecideRequest(BaseModel):
    state: object = Field(..., description="Any state: text, dict of fields.")
    questions: dict
    threshold: float = 0.85


@app.post("/decide")
def decide(req: DecideRequest):
    answers = agent.decide(req.state, req.questions)
    gated = gate(answers, threshold=req.threshold)
    return {
        "answers": answers,
        "gating": {name: {"confidence_threshold": req.threshold, "automatable": ok}
                   for name, _, ok in gated},
    }


@app.get("/health")
def health():
    return {"status": "ok", "checkpoint": agent.model.cfg if agent else None}


# ---- Jev-compatible endpoint (TypeSafe System One schema) ----

class SystemOneRequest(BaseModel):
    model: str = "oev"
    state: object = Field(..., description="Any state: text, dict of fields.")
    questions: dict


def _jevify(answers: dict) -> dict:
    """Map OEV answers onto the Jev System One response shape."""
    out = {}
    for name, a in answers.items():
        if isinstance(a, dict) and "probabilities" in a:
            choice = a.get("choice") or a.get("value")
            out[name] = {
                "answer": choice,
                "confidence": a.get("confidence"),
                "probabilities": a["probabilities"],
            }
        else:
            # noul returns a probability directly
            out[name] = {"answer": bool(a >= 0.5), "confidence": a if a >= 0.5 else 1 - a,
                         "probabilities": {"yes": a, "no": 1 - a}}
    return out


@app.post("/v1/systemone")
def systemone(req: SystemOneRequest):
    """Jev-compatible endpoint: existing TypeSafe clients work by changing baseUrl."""
    answers = agent.decide(req.state, req.questions)
    return {
        "model": req.model,
        "answers": _jevify(answers),
        "routing": {"model": "oev", "checkpoint": "local"},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    global agent
    agent = OEV(args.checkpoint, device=args.device)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
