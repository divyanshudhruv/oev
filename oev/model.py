from dataclasses import dataclass
import torch
import torch.nn as nn
from oev.tokenizer import VOCAB_SIZE, PAD_ID


@dataclass
class OEVConfig:
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    vocab_size: int = VOCAB_SIZE
    max_len: int = 512
    dropout: float = 0.1


PRESETS = {
    "tiny": {"d_model": 96, "n_layers": 2, "n_heads": 4, "d_ff": 384},
    "base": {"d_model": 384, "n_layers": 6, "n_heads": 6, "d_ff": 1536},
}


class OEVModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=PAD_ID)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            cfg.d_model,
            cfg.n_heads,
            cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, 1)

    def forward(self, ids, pad_mask, anchor_pos):
        L = ids.size(1)
        x = self.tok(ids) + self.pos(torch.arange(L, device=ids.device))
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.norm(x)
        h = x.gather(1, anchor_pos.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        return self.head(h).squeeze(-1)


class HFBackboneOEV(nn.Module):
    def __init__(self, backbone="microsoft/deberta-v3-small", dropout=0.1):
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(backbone)
        d = self.backbone.config.hidden_size
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def forward(self, ids, pad_mask, anchor_pos):
        x = self.backbone(input_ids=ids, attention_mask=~pad_mask).last_hidden_state
        x = self.norm(x)
        h = x.gather(1, anchor_pos.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        return self.head(h).squeeze(-1)
