import math

import torch
import torch.nn.functional as F

from oev.dataset import pack
from oev.evaluate import load_model
from oev.tokenizer_hf import HFTokenPacker


def normalize_question(name, q):
    """Accept either question schema and return an OEV-schema copy.

    Native schema: choice uses a bare "options" list, score uses "levels".
    Jev / TypeSafe schema: choice and noul carry "criteria" as a map of
    option name -> description, score carries "criteria" as a list of level
    labels. Descriptions are ignored (OEV was trained on bare options); the
    option names become the options. Native keys win when both are present.
    """
    q = dict(q)
    criteria = q.get("criteria")
    if q.get("type") == "choice" and "options" not in q:
        if isinstance(criteria, dict) and criteria:
            q["options"] = [str(option) for option in criteria]
        else:
            raise ValueError(
                f"question {name!r} (choice) needs an 'options' list "
                "or a 'criteria' map of option name -> description"
            )
    elif q.get("type") == "score" and "levels" not in q:
        if isinstance(criteria, list) and criteria:
            q["levels"] = list(criteria)
        else:
            raise ValueError(
                f"question {name!r} (score) needs a 'levels' list "
                "or a 'criteria' list of level labels"
            )
    return q


class OEV:
    def __init__(self, checkpoint="checkpoints/oev-tiny.pt", device="cpu", temperature=1.0):
        self.model = load_model(checkpoint, device)
        self.device = device
        self.temperature = float(temperature)
        if hasattr(self.model.cfg, "max_len"):
            self.max_len = self.model.cfg.max_len
            self.packer = None
        else:
            self.max_len = self.model.cfg.get("max_len", 512)
            self.packer = HFTokenPacker(self.model.cfg["backbone"])

    def _question(self, name, q):
        q = normalize_question(name, q)
        pq = {"name": name, "type": q["type"], "instructions": q.get("instructions", q["type"])}
        if q["type"] == "choice":
            pq["options"] = list(q["options"])
        elif q["type"] == "noul":
            pq["options"] = ["no", "yes"]
        else:
            pq["options"] = [str(level) for level in q["levels"]]
        pq["answer"] = pq["options"][0]
        return pq

    def _probs(self, state, pq, temperature=None):
        temperature = self.temperature if temperature is None else float(temperature)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be a finite positive number")
        if self.packer is not None:
            ids, anchors, _ = self.packer.pack(state, pq, self.max_len)
        else:
            ids, anchors, _ = pack(state, pq, self.max_len)
        t = torch.tensor([ids], dtype=torch.long)
        pad = torch.zeros(1, len(ids), dtype=torch.bool)
        apos = torch.tensor([anchors], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(t.to(self.device), pad.to(self.device), apos.to(self.device))
        return F.softmax(logits / temperature, dim=-1)[0].tolist()

    def decide(self, state, questions, temperature=None):
        out = {}
        for name, raw_question in questions.items():
            q = normalize_question(name, raw_question)
            pq = self._question(name, q)
            probs = self._probs(state, pq, temperature=temperature)
            best = max(range(len(probs)), key=probs.__getitem__)
            if q["type"] == "noul":
                out[name] = float(probs[1])
            elif q["type"] == "score":
                levels = q["levels"]
                out[name] = {
                    "value": levels[best],
                    "probabilities": {levels[i]: float(p) for i, p in enumerate(probs)},
                }
            else:
                out[name] = {
                    "choice": pq["options"][best],
                    "probabilities": {o: float(p) for o, p in zip(pq["options"], probs)},
                    "confidence": float(max(probs)),
                }
        return out
