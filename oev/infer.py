import math

import torch
import torch.nn.functional as F

from oev.dataset import pack
from oev.evaluate import load_model
from oev.tokenizer_hf import HFTokenPacker


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
        for name, q in questions.items():
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
