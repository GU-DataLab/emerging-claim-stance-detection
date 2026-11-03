"""
Stance detection models (HF-compatible).

- CrossAttentionModel: PLM encoder + two-way claim<->post cross-attention
  (+ optional supervised-contrastive projection head). This is the proposed model.
- BaselineModel: PLM encoder + mean pooling (+ optional projection head).
  Used for the fine-tuned PLM baseline and the "w/o XAttn" ablation.

Both models are saved/loaded with `save_pretrained` / `from_pretrained`.
To start training from a pretrained PLM, use `Model.from_plm(config)`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.utils import ModelOutput


class StanceConfig(PretrainedConfig):
    model_type = "stance"

    def __init__(
        self,
        model_name: str = "vinai/bertweet-large",
        num_classes: int = 3,
        label2id: Optional[dict] = None,
        id2label: Optional[dict] = None,
        cross_attn_type: str = "two-way",  # "one-way" | "two-way"
        num_heads: int = 8,
        pool_hidden: bool = False,
        use_cl: bool = False,
        cl_dim: int = 128,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.num_classes = num_classes
        self.label2id = label2id or {"Refute": 0, "Support": 1, "Neutral": 2}
        self.id2label = id2label or {v: k for k, v in self.label2id.items()}
        self.cross_attn_type = cross_attn_type
        self.num_heads = num_heads
        self.pool_hidden = pool_hidden
        self.use_cl = use_cl
        self.cl_dim = cl_dim
        self.dropout = dropout


@dataclass
class StanceModelOutput(ModelOutput):
    logits: torch.Tensor = None
    embeddings: Optional[torch.Tensor] = None   # pooled claim-post representation
    projections: Optional[torch.Tensor] = None  # L2-normalized z = Proj(r), only if use_cl


def split_joint_embeddings(
    tokenizer,
    hidden: torch.Tensor,          # (B, L, d)
    input_ids: torch.Tensor,       # (B, L)
    attention_mask: torch.Tensor,  # (B, L)
):
    """
    Split the encoded pair back into claim tokens and post tokens.

    Works with either layout:
        A) <s> claim </s> post </s>
        B) <s> claim </s> </s> post </s>   (double-SEP, RoBERTa/BART pair encoding)

    Returns claim_tok (B, Lc, d), post_tok (B, Lp, d), claim_mask (B, Lc), post_mask (B, Lp)
    where masks are True for real tokens.
    """
    sep_id = tokenizer.sep_token_id
    B, _, d = hidden.size()
    device = hidden.device
    sep_positions = input_ids == sep_id

    spans = []
    for i in range(B):
        seps = sep_positions[i].nonzero(as_tuple=False).flatten()
        first_sep, last_sep = seps[0].item(), seps[-1].item()
        c_len = first_sep - 1  # claim = tokens [1, first_sep)
        p_start = first_sep + 1
        if len(seps) >= 2 and seps[1].item() == first_sep + 1:
            p_start += 1  # skip extra SEP
        p_len = max(0, last_sep - p_start)
        spans.append((c_len, p_start, p_len))

    Lc = max(s[0] for s in spans)
    Lp = max(s[2] for s in spans)
    claim_tok = torch.zeros(B, Lc, d, device=device, dtype=hidden.dtype)
    post_tok = torch.zeros(B, Lp, d, device=device, dtype=hidden.dtype)
    claim_mask = torch.zeros(B, Lc, dtype=torch.bool, device=device)
    post_mask = torch.zeros(B, Lp, dtype=torch.bool, device=device)

    for i, (c_len, p_start, p_len) in enumerate(spans):
        if c_len > 0:
            claim_tok[i, :c_len] = hidden[i, 1 : 1 + c_len]
            claim_mask[i, :c_len] = True
        if p_len > 0:
            post_tok[i, :p_len] = hidden[i, p_start : p_start + p_len]
            post_mask[i, :p_len] = True
    return claim_tok, post_tok, claim_mask, post_mask


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).float()
    return (x * m).sum(1) / m.sum(1).clamp(min=1e-9)


def _build_plm(model_name: str, pretrained: bool):
    if pretrained:
        try:
            return AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        except TypeError:
            return AutoModel.from_pretrained(model_name)
    plm_cfg = AutoConfig.from_pretrained(model_name)
    try:
        return AutoModel.from_config(plm_cfg, add_pooling_layer=False)
    except TypeError:
        return AutoModel.from_config(plm_cfg)


def _mlp(hid: int, out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hid, hid // 2),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hid // 2, out),
    )


class _BaseStanceModel(PreTrainedModel):
    config_class = StanceConfig
    base_model_prefix = "plm"

    def __init__(self, config: StanceConfig):
        super().__init__(config)
        self._tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        # randomly initialised here; `from_plm` swaps in pretrained weights and
        # `from_pretrained` overwrites everything with the checkpoint.
        self.plm = _build_plm(config.model_name, pretrained=False)
        hid = self.plm.config.hidden_size
        self.classifier = _mlp(hid, config.num_classes, config.dropout)
        if config.use_cl:
            self.projector = _mlp(hid, config.cl_dim, config.dropout)
        self._build_extra_layers(config, hid)
        self.post_init()

    def _build_extra_layers(self, config: StanceConfig, hid: int):
        pass

    # --- construction / persistence ---------------------------------------
    @classmethod
    def from_plm(cls, config: StanceConfig):
        """New model whose encoder starts from the pretrained `config.model_name` weights."""
        model = cls(config)
        model.plm = _build_plm(config.model_name, pretrained=True)
        return model

    @classmethod
    def from_pretrained(cls, pretrained_path: str, **kwargs):
        model = super().from_pretrained(pretrained_path, **kwargs)
        model._tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        return model

    def save_pretrained(self, save_directory: str, **kwargs):
        super().save_pretrained(save_directory, **kwargs)
        self.tokenizer.save_pretrained(save_directory)

    def _init_weights(self, module):
        # keep PyTorch default init for the new layers; PLM weights come from from_plm/from_pretrained
        pass

    # --- helpers ------------------------------------------------------------
    @property
    def tokenizer(self):
        return self._tokenizer

    def get_input_embeddings(self):
        return self.plm.get_input_embeddings()

    def head_parameters(self):
        """Parameters of the randomly initialised layers (trained with the higher learning rate)."""
        return [p for n, p in self.named_parameters() if not n.startswith("plm.")]

    def encode_pair(self, claims: List[str], posts: List[str], max_len: int = 256):
        text = [f"{c} {self.tokenizer.sep_token} {p}" for c, p in zip(claims, posts)]
        return self.tokenizer(
            text, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
        ).to(self.device)

    def _encode(self, input_ids, attention_mask, token_type_ids, position_ids):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if "bart" in self.config.model_name.lower():  # BART has no token_type_ids
            out = self.plm(input_ids=input_ids, attention_mask=attention_mask)
        else:
            if token_type_ids is not None and self.plm.config.type_vocab_size == 1:
                token_type_ids = torch.zeros_like(token_type_ids)  # RoBERTa ignores segments
            out = self.plm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
            )
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return hidden, attention_mask

    def _pool(self, hidden, input_ids, attention_mask) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        *,
        claims: Optional[List[str]] = None,
        posts: Optional[List[str]] = None,
        return_dict: bool = True,
    ):
        if input_ids is None:
            assert claims is not None and posts is not None, "pass input_ids or claims+posts"
            batch = self.encode_pair(claims, posts)
            input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
            token_type_ids = batch.get("token_type_ids")

        hidden, attention_mask = self._encode(input_ids, attention_mask, token_type_ids, position_ids)
        pooled = self._pool(hidden, input_ids, attention_mask)
        logits = self.classifier(pooled)
        proj = F.normalize(self.projector(pooled), dim=1) if self.config.use_cl else None

        output = StanceModelOutput(logits=logits, embeddings=pooled, projections=proj)
        return output if return_dict else output.to_tuple()


class BaselineModel(_BaseStanceModel):
    """Fine-tuned PLM: masked mean pooling over the whole [claim; post] sequence."""

    def _pool(self, hidden, input_ids, attention_mask):
        return masked_mean(hidden, attention_mask)


class CrossAttentionModel(_BaseStanceModel):
    """PLM + two-way claim<->post cross-attention."""

    def _build_extra_layers(self, config: StanceConfig, hid: int):
        # NOTE: a single MultiheadAttention module is used for both directions
        # (claim->post and post->claim), i.e. the two blocks share weights.
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hid, num_heads=config.num_heads, batch_first=True, dropout=0.05
        )
        self.post_attn = nn.Sequential(nn.LayerNorm(hid), nn.Dropout(0.05))

    def _pool(self, hidden, input_ids, attention_mask):
        c_tok, p_tok, c_mask, p_mask = split_joint_embeddings(
            self.tokenizer, hidden, input_ids, attention_mask
        )
        if self.config.cross_attn_type == "two-way":
            # claim queries post, post queries claim, residual + LayerNorm
            attn_c, _ = self.cross_attention(query=c_tok, key=p_tok, value=p_tok, key_padding_mask=~p_mask)
            attn_p, _ = self.cross_attention(query=p_tok, key=c_tok, value=c_tok, key_padding_mask=~c_mask)
            h_c = self.post_attn(c_tok + attn_c)
            h_p = self.post_attn(p_tok + attn_p)
            # masked mean pooling, then token-count weighted average
            l_c = c_mask.sum(1, keepdim=True).float()
            l_p = p_mask.sum(1, keepdim=True).float()
            pooled = (masked_mean(h_c, c_mask) * l_c + masked_mean(h_p, p_mask) * l_p) / (l_c + l_p).clamp(min=1e-9)
        else:  # one-way: claim queries post only
            attn_c, _ = self.cross_attention(query=c_tok, key=p_tok, value=p_tok, key_padding_mask=~p_mask)
            pooled = masked_mean(self.post_attn(c_tok + attn_c), c_mask)

        if self.config.pool_hidden:
            pooled = self.post_attn(pooled + masked_mean(hidden, attention_mask))
        return pooled


MODEL_CLASSES = {"baseline": BaselineModel, "cross_attention": CrossAttentionModel}


def load_model(ckpt_dir: str, device: str = "cpu"):
    """Load a saved checkpoint; the model class is inferred from the saved weights."""
    import json
    import os

    with open(os.path.join(ckpt_dir, "config.json")) as f:
        archs = json.load(f).get("architectures") or ["CrossAttentionModel"]
    cls = BaselineModel if archs[0] == "BaselineModel" else CrossAttentionModel
    model = cls.from_pretrained(ckpt_dir)
    model.config.id2label = {int(k): v for k, v in model.config.id2label.items()}
    return model.to(device).eval()
