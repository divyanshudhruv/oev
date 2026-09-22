from transformers import AutoTokenizer, PreTrainedTokenizerFast


class HFTokenPacker:
    def __init__(self, name):
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
        ids += self.tok.encode(state, add_special_tokens=False)[: max_len // 2]
        ids += [self.tok.sep_token_id or self.tok.eos_token_id]
        ids += self.tok.encode(q_text, add_special_tokens=False)
        anchor_pos = []
        for o in options:
            anchor_pos.append(len(ids))
            ids += [self.anchor_id] + self.tok.encode(" " + o, add_special_tokens=False)
        ids = ids[:max_len]
        label = options.index(str(question["answer"]))
        return ids, anchor_pos, label
