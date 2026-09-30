"""Minimal HTTP server for OEV decisions.

    python -m oev.serve --checkpoint checkpoints_td5/oev-tiny.pt --port 8000
    python -m oev.serve --onnx exported.int8.onnx

/decide speaks the native schema; /v1/systemone accepts the Jev/TypeSafe
criteria schema, so existing clients work by changing baseUrl.
"""

import argparse
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from oev.evaluate import pack_question
from oev.presets import gate

app = FastAPI(title="OEV", description="System One decision model: typed questions in, calibrated probabilities out.")

# set in main(): TorchBackend (checkpoint) or OnnxBackend (int8 export)
agent = None


class DecideRequest(BaseModel):
    state: object = Field(..., description="Any state: text, dict of fields.")
    questions: dict
    threshold: float = 0.85


@app.post("/decide")
def decide(req: DecideRequest):
    if agent is None:
        raise HTTPException(status_code=503, detail="no checkpoint loaded - run oev-serve --checkpoint ...")
    answers = agent.decide(req.state, req.questions)
    gated = gate(answers, threshold=req.threshold)
    return {
        "answers": answers,
        "gating": {name: {"confidence_threshold": req.threshold, "automatable": ok}
                   for name, _, ok in gated},
    }


@app.get("/health")
def health():
    if agent is None:
        # 503 (not 200) so Docker HEALTHCHECK and orchestrators can tell
        # "still loading" apart from "failed to load" during startup
        raise HTTPException(status_code=503, detail="model not loaded yet")
    return {"status": "ok", **agent.describe()}


# Jev-compatible endpoint (TypeSafe System One schema)

class SystemOneRequest(BaseModel):
    model: str = "oev"
    state: object = Field(..., description="Any state: text, dict of fields.")
    questions: dict


def _jevify(answers: dict) -> dict:
    # map OEV answers onto the Jev System One response shape
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


def _validate_jev_questions(questions: dict) -> str | None:
    # return an error string for anything the Jev schema does not allow
    for name, question in questions.items():
        if not isinstance(question, dict):
            return f"question {name} must be an object"
        qtype = question.get("type")
        if qtype not in {"choice", "noul", "score"}:
            return f"question {name} has unsupported type"
        if qtype in {"choice", "noul"}:
            criteria = question.get("criteria")
            if criteria is not None and (not isinstance(criteria, dict) or not criteria):
                return f"question {name} ({qtype}) criteria must be a non-empty object"
        else:
            criteria = question.get("criteria")
            if not isinstance(criteria, list) or not criteria:
                return f"question {name} (score) needs a non-empty criteria list of level labels"
    return None


@app.post("/v1/systemone")
def systemone(req: SystemOneRequest):
    # drop-in for TypeSafe clients: same request/response shape as Jev
    if agent is None:
        raise HTTPException(status_code=503, detail="no checkpoint loaded - run oev-serve --checkpoint ...")
    error = _validate_jev_questions(req.questions)
    if error:
        raise HTTPException(status_code=422, detail=error)
    answers = agent.decide(req.state, req.questions)
    return {
        "model": req.model,
        "answers": _jevify(answers),
        "routing": {"model": "oev", "checkpoint": "local"},
    }


class TorchBackend:
    # thin adapter so /health works the same for both backends
    def __init__(self, checkpoint, device="cpu"):
        from oev.infer import OEV

        self.inner = OEV(checkpoint, device=device)

    def decide(self, state, questions):
        return self.inner.decide(state, questions)

    def describe(self):
        return {"backend": "torch", "checkpoint": self.inner.model.cfg}


class OnnxBackend:
    # torch-free mirror of oev.infer.OEV.decide: pack -> logits -> softmax.
    # The question normalization is duplicated here on purpose: oev.infer
    # imports torch at module level and this backend must not.
    def __init__(self, path, backbone="microsoft/deberta-v3-base", max_len=768, threads=None):
        import numpy as np
        import onnxruntime as ort

        self.np = np
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
        self.path = path
        self.backbone = backbone
        self.max_len = max_len
        from oev.tokenizer_hf import HFTokenPacker

        self.packer = HFTokenPacker(backbone)

    def describe(self):
        return {"backend": "onnx", "checkpoint": self.path,
                "backbone": self.backbone, "max_len": self.max_len}

    @staticmethod
    def _normalize_question(name, q):
        q = dict(q)
        criteria = q.get("criteria")
        if q.get("type") == "choice" and "options" not in q:
            if isinstance(criteria, dict) and criteria:
                q["options"] = [str(option) for option in criteria]
            else:
                raise ValueError(f"question {name!r} (choice) needs options or a criteria map")
        elif q.get("type") == "score" and "levels" not in q:
            if isinstance(criteria, list) and criteria:
                q["levels"] = list(criteria)
            else:
                raise ValueError(f"question {name!r} (score) needs levels or a criteria list")
        return q

    def _probs(self, state, pq):
        ids, anchors, _ = self.packer.pack(state, pq, self.max_len)
        np = self.np
        feed = {
            "ids": np.asarray([ids], dtype=np.int64),
            "pad_mask": np.zeros((1, len(ids)), dtype=bool),
            "anchor_pos": np.asarray([anchors], dtype=np.int64),
        }
        logits = self.sess.run(None, feed)[0][0]
        top = logits.max()
        exp = np.exp(logits - top)
        return (exp / exp.sum()).tolist()

    def decide(self, state, questions):
        out = {}
        for name, raw in questions.items():
            q = self._normalize_question(name, raw)
            if q["type"] == "noul":
                options = ["no", "yes"]
            elif q["type"] == "choice":
                options = [str(option) for option in q["options"]]
            else:
                options = [str(level) for level in q["levels"]]
            pq = pack_question(dict(q, name=name, options=options,
                                    answer=options[0]))
            probs = self._probs(state, pq)
            best = max(range(len(probs)), key=probs.__getitem__)
            if q["type"] == "noul":
                out[name] = float(probs[1])
            elif q["type"] == "score":
                levels = [str(level) for level in q["levels"]]
                out[name] = {"value": levels[best],
                             "probabilities": {levels[i]: float(p) for i, p in enumerate(probs)}}
            else:
                out[name] = {"choice": pq["options"][best],
                             "probabilities": {o: float(p) for o, p in zip(pq["options"], probs)},
                             "confidence": float(max(probs))}
        return out


def _resolve_checkpoint(spec: str) -> str:
    # accept a local path or a Hugging Face id (repo/filename)
    if "/" not in spec or os.path.exists(spec):
        return spec
    from huggingface_hub import hf_hub_download

    repo, _, filename = spec.rpartition("/")
    return hf_hub_download(repo_id=repo, filename=filename)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=os.environ.get("OEV_CHECKPOINT", ""),
                   help="path to a .pt checkpoint or a Hugging Face id (repo/filename); OEV_CHECKPOINT env works too")
    p.add_argument("--onnx", default=os.environ.get("OEV_ONNX", ""),
                   help="path to an exported ONNX graph (int8 recommended); torch-free backend, OEV_ONNX env works too")
    p.add_argument("--backbone", default="microsoft/deberta-v3-base",
                   help="ONNX backend only: tokenizer/backbone used at export time")
    p.add_argument("--max-len", type=int, default=768,
                   help="ONNX backend only: max_len the checkpoint was trained with")
    p.add_argument("--threads", type=int, default=None,
                   help="ONNX backend only: intra-op thread count (default: onnxruntime decides)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if not args.checkpoint and not args.onnx:
        p.error("--checkpoint (or OEV_CHECKPOINT) or --onnx (or OEV_ONNX) is required")
    global agent
    if args.onnx:
        agent = OnnxBackend(args.onnx, backbone=args.backbone, max_len=args.max_len, threads=args.threads)
    else:
        agent = TorchBackend(_resolve_checkpoint(args.checkpoint), device=args.device)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

