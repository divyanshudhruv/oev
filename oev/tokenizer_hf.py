class HFTokenPacker:
    def __init__(self, name):
        from transformers import AutoTokenizer, PreTrainedTokenizerFast
        self.name = name
        try:
            self.tok = AutoTokenizer.from_pretrained(name)
        except (OSError, ValueError):
            self.tok = PreTrainedTokenizerFast.from_pretrained(name)
        self.tok.add_tokens(["[ANCHOR]"])
        self.anchor_id = self.tok.convert_tokens_to_ids("[ANCHOR]")

    def pack(self, state, question, max_len=512):
        options = question["options"]
        q_text = question["name"] + ": " + question.get("instructions", question["type"])
        ids = [self.tok.cls_token_id or self.tok.bos_token_id]
        opt_ids = [[self.anchor_id] + self.tok.encode(" " + o, add_special_tokens=False) for o in options]
        q_ids = self.tok.encode(q_text, add_special_tokens=False)
        fixed = 1 + 1 + len(q_ids) + sum(len(o) for o in opt_ids)
        budget = max(1, max_len - fixed)
        ids += self.tok.encode(state, add_special_tokens=False)[:budget]
        ids += [self.tok.sep_token_id or self.tok.eos_token_id]
        ids += q_ids
        anchor_pos = []
        for o in opt_ids:
            anchor_pos.append(len(ids))
            ids += o
        ids = ids[:max_len]
        # anchors were recorded pre-clip; when fixed overhead exceeds max_len the
        # tail gets clipped and stale anchors can point past the sequence -> clamp
        anchor_pos = [min(a, len(ids) - 1) for a in anchor_pos]
        label = options.index(str(question["answer"]))
        return ids, anchor_pos, label
