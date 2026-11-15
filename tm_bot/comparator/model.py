"""Independently designed Deep Sets card encoder and per-candidate MLP scorer.

The comparator shares the pure observation/candidate *encoding functions* with the
main agent (`tm_bot.models.encode`) but none of its modules, weights, or state.
Its card encoder is a permutation-invariant Deep Sets network (masked mean + max
pooling) rather than the main agent's transformer, and each candidate is scored by
an MLP conditioned on the pooled observation and the choice context.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from tm_bot.models.encode import (
    CAND_FEAT_DIM,
    CARD_DYN_DIM,
    CHOICE_CTX_DIM,
    OBS_SCALAR_DIM,
    SPACE_FEAT_DIM,
)
from tm_bot.models.vocab import (
    CARD_STATIC_DIM,
    PAD_ID,
    PROMPT_TYPES,
    SLOT_NAMES,
    CardVocab,
)

COMPARATOR_ARCHITECTURE = "comparator_deepset_v1"


@dataclass(frozen=True)
class ComparatorConfig:
    """Architecture and vocabulary configuration for the comparator network."""

    vocab_size: int
    card_static_dim: int = CARD_STATIC_DIM
    obs_scalar_dim: int = OBS_SCALAR_DIM
    space_feat_dim: int = SPACE_FEAT_DIM
    card_dyn_dim: int = CARD_DYN_DIM
    choice_ctx_dim: int = CHOICE_CTX_DIM
    cand_feat_dim: int = CAND_FEAT_DIM
    num_slots: int = len(SLOT_NAMES)
    num_prompt_types: int = len(PROMPT_TYPES)
    d_model: int = 384
    card_hidden: int = 384
    card_emb_dim: int = 128
    card_static_proj_dim: int = 128
    slot_emb_dim: int = 32
    card_dyn_proj_dim: int = 32
    space_hidden: int = 96
    choice_hidden: int = 256
    cand_hidden: int = 384
    value_hidden: int = 384
    dropout: float = 0.05
    architecture: str = COMPARATOR_ARCHITECTURE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ComparatorConfig:
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in payload.items() if k in valid})


def _masked_mean_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def _masked_max_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    has_any = mask.any(dim=1, keepdim=True)
    x_masked = x.masked_fill(~mask.unsqueeze(-1), -1e4)
    pooled = x_masked.max(dim=1).values
    return torch.where(has_any, pooled, torch.zeros_like(pooled))


class ComparatorNetwork(nn.Module):
    """Deep Sets observation/card encoder with a candidate scorer and win-probability head."""

    def __init__(self, config: ComparatorConfig, card_static_matrix: torch.Tensor) -> None:
        super().__init__()
        if card_static_matrix.shape != (config.vocab_size, config.card_static_dim):
            raise ValueError(
                f"Expected card_static_matrix shape "
                f"{(config.vocab_size, config.card_static_dim)}, got "
                f"{tuple(card_static_matrix.shape)}"
            )
        self.config = config
        self.register_buffer(
            "card_static_matrix",
            card_static_matrix.detach().clone().to(dtype=torch.float32),
        )

        # Shared card representation (observation tokens and candidate tokens).
        self.card_id_emb = nn.Embedding(
            config.vocab_size, config.card_emb_dim, padding_idx=PAD_ID
        )
        self.card_static_proj = nn.Linear(
            config.card_static_dim, config.card_static_proj_dim
        )
        self.slot_emb = nn.Embedding(
            config.num_slots, config.slot_emb_dim, padding_idx=PAD_ID
        )
        self.card_dyn_proj = nn.Linear(config.card_dyn_dim, config.card_dyn_proj_dim)

        card_in = (
            config.card_emb_dim
            + config.card_static_proj_dim
            + config.slot_emb_dim
            + config.card_dyn_proj_dim
        )
        h = config.card_hidden
        self.card_set_mlp = nn.Sequential(
            nn.Linear(card_in, h),
            nn.LayerNorm(h),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(h, h),
            nn.SiLU(),
        )
        self.scalar_mlp = nn.Sequential(
            nn.Linear(config.obs_scalar_dim, h),
            nn.LayerNorm(h),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(h, h),
            nn.SiLU(),
        )
        sh = config.space_hidden
        self.space_mlp = nn.Sequential(
            nn.Linear(config.space_feat_dim, sh),
            nn.LayerNorm(sh),
            nn.SiLU(),
            nn.Linear(sh, sh),
            nn.SiLU(),
        )
        fused_in = h + 2 * h + 2 * sh
        d = config.d_model
        self.obs_fuse = nn.Sequential(
            nn.Linear(fused_in, d),
            nn.LayerNorm(d),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )

        # Win-probability head over the pooled observation.
        vh = config.value_hidden
        self.value_head = nn.Sequential(
            nn.Linear(d, vh),
            nn.LayerNorm(vh),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(vh, vh),
            nn.SiLU(),
            nn.Linear(vh, 1),
        )

        # Typed-candidate scorer: choice context + candidate MLP conditioned on the observation.
        ch = config.choice_hidden
        self.prompt_emb = nn.Embedding(config.num_prompt_types, 32)
        self.picked_proj = nn.Linear(
            config.card_emb_dim + config.card_static_proj_dim, 128
        )
        self.choice_mlp = nn.Sequential(
            nn.Linear(config.choice_ctx_dim + 32 + 128, ch),
            nn.LayerNorm(ch),
            nn.SiLU(),
            nn.Linear(ch, ch),
        )
        cand_feat_hidden = 128
        self.cand_feat_proj = nn.Linear(config.cand_feat_dim, cand_feat_hidden)
        cand_in = config.card_emb_dim + config.card_static_proj_dim + cand_feat_hidden
        chd = config.cand_hidden
        self.cand_mlp = nn.Sequential(
            nn.Linear(cand_in, chd),
            nn.LayerNorm(chd),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(chd, chd),
            nn.SiLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(d + ch + chd, chd),
            nn.LayerNorm(chd),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(chd, chd),
            nn.SiLU(),
            nn.Linear(chd, 1),
        )

    def _embed_cards(self, card_ids: torch.Tensor) -> torch.Tensor:
        id_vec = self.card_id_emb(card_ids)
        static_vec = self.card_static_proj(
            F.embedding(card_ids, self.card_static_matrix)
        )
        return torch.cat([id_vec, static_vec], dim=-1)

    def encode_observations(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return the pooled observation embedding `[B, D]`."""
        card_ids = batch["card_ids"]
        card_mask = batch["card_mask"].bool()
        card_h = self.card_set_mlp(
            torch.cat(
                [
                    self._embed_cards(card_ids),
                    self.slot_emb(batch["card_slots"]),
                    self.card_dyn_proj(batch["card_dyn"]),
                ],
                dim=-1,
            )
        )
        card_pool = torch.cat(
            [
                _masked_mean_pool(card_h, card_mask),
                _masked_max_pool(card_h, card_mask),
            ],
            dim=-1,
        )
        scalars = self.scalar_mlp(batch["obs_scalars"])
        space_h = self.space_mlp(batch["space_feats"])
        space_mask = batch["space_mask"].bool()
        space_pool = torch.cat(
            [
                _masked_mean_pool(space_h, space_mask),
                _masked_max_pool(space_h, space_mask),
            ],
            dim=-1,
        )
        return self.obs_fuse(torch.cat([scalars, card_pool, space_pool], dim=-1))

    def forward_value(self, obs_pooled: torch.Tensor) -> torch.Tensor:
        """Return the raw win-probability logit `[B]`."""
        return self.value_head(obs_pooled).squeeze(-1)

    def forward_policy(
        self, obs_pooled: torch.Tensor, batch: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score every candidate conditioned on the observation and choice context.

        Returns `(masked_logits, probs)` with `-inf` logits and exact `0.0`
        probability for illegal/padded candidates.
        """
        cand_mask = batch["cand_mask"].bool()
        prompt_vec = self.prompt_emb(batch["prompt_type_ids"])
        picked_h = self._masked_picked(batch["picked_card_ids"], batch["picked_mask"])
        choice = self.choice_mlp(
            torch.cat([batch["choice_ctx"], prompt_vec, picked_h], dim=-1)
        )

        cand = self.cand_mlp(
            torch.cat(
                [
                    self._embed_cards(batch["cand_card_ids"]),
                    self.cand_feat_proj(batch["cand_feats"]),
                ],
                dim=-1,
            )
        )
        bsz, n_cand, _ = cand.shape
        obs_rep = obs_pooled.unsqueeze(1).expand(bsz, n_cand, -1)
        choice_rep = choice.unsqueeze(1).expand(bsz, n_cand, -1)
        raw_logits = (
            self.scorer(torch.cat([obs_rep, choice_rep, cand], dim=-1))
            .squeeze(-1)
            .float()
        )
        masked_logits = raw_logits.masked_fill(~cand_mask, float("-inf"))

        has_cand = cand_mask.any(dim=1)
        probs = torch.zeros_like(masked_logits)
        if has_cand.any():
            probs[has_cand] = F.softmax(masked_logits[has_cand], dim=-1)
        probs = torch.where(cand_mask, probs, torch.zeros_like(probs))
        return masked_logits, probs

    def _masked_picked(self, picked_ids: torch.Tensor, picked_mask: torch.Tensor) -> torch.Tensor:
        mask = picked_mask.bool()
        base = self._embed_cards(picked_ids)
        return self.picked_proj(_masked_mean_pool(base, mask))

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        compute_policy: bool = True,
        compute_value: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Forward pass compatible with the shared calibration/entropy helpers."""
        obs_pooled = self.encode_observations(batch)
        out: dict[str, torch.Tensor] = {"obs_pooled": obs_pooled}
        if compute_value:
            value_logit = self.forward_value(obs_pooled).float()
            win_prob = torch.sigmoid(value_logit)
            out["value_logit"] = value_logit
            out["win_prob"] = win_prob
            out["value"] = 2.0 * win_prob - 1.0
        if compute_policy and "cand_card_ids" in batch:
            logits, probs = self.forward_policy(obs_pooled, batch)
            out["policy_logits"] = logits
            out["policy_probs"] = probs
        return out

    def count_parameters(self, *, trainable_only: bool = False) -> int:
        params = self.parameters() if not trainable_only else (
            p for p in self.parameters() if p.requires_grad
        )
        return sum(p.numel() for p in params)

    @classmethod
    def from_vocab(
        cls,
        vocab: CardVocab,
        config: ComparatorConfig | None = None,
    ) -> ComparatorNetwork:
        cfg = config if config is not None else ComparatorConfig(vocab_size=vocab.size)
        if cfg.vocab_size != vocab.size:
            raise ValueError(
                f"ComparatorConfig.vocab_size ({cfg.vocab_size}) != vocab.size ({vocab.size})"
            )
        static_mat = torch.from_numpy(vocab.static_features)
        return cls(config=cfg, card_static_matrix=static_mat)


def save_comparator_checkpoint(
    path: str | Path,
    model: ComparatorNetwork,
    vocab: CardVocab,
    *,
    temperature: float = 1.0,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save comparator weights, architecture, vocab, calibration, and parameter count."""
    from tm_bot.models.network import save_checkpoint

    meta = dict(metadata or {})
    meta["temperature"] = float(temperature)
    meta["architecture"] = COMPARATOR_ARCHITECTURE
    return save_checkpoint(path, model, vocab, metadata=meta)  # type: ignore[arg-type]


def load_comparator_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[ComparatorNetwork, CardVocab, dict[str, Any]]:
    """Load `(model, vocab, metadata)` saved by `save_comparator_checkpoint`."""
    target = Path(path)
    ckpt_path = target / "checkpoint.pt" if target.is_dir() else target
    payload = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    cfg = ComparatorConfig.from_dict(payload["config"])
    if cfg.architecture != COMPARATOR_ARCHITECTURE:
        raise ValueError(
            f"{ckpt_path} is a {cfg.architecture!r} checkpoint, not a comparator checkpoint"
        )
    vocab = CardVocab.from_dict(payload["vocab"])
    model = ComparatorNetwork.from_vocab(vocab, cfg)
    model.load_state_dict(payload["model_state_dict"])
    meta = dict(payload.get("metadata", {}))
    model.temperature = float(meta.get("temperature", 1.0))
    model = model.to(map_location)
    model.eval()
    return model, vocab, meta


def comparator_checkpoint_architecture(path: str | Path) -> str:
    """Return the architecture string recorded in a checkpoint payload."""
    target = Path(path)
    ckpt_path = target / "checkpoint.pt" if target.is_dir() else target
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return str(payload.get("config", {}).get("architecture", ""))


def parameter_count_report(model: ComparatorNetwork) -> dict[str, Any]:
    """Small helper used by docs/tests to report model size."""
    return {
        "parameter_count": model.count_parameters(),
        "trainable_parameter_count": model.count_parameters(trainable_only=True),
        "config": model.config.to_dict(),
    }


def write_parameter_config(model: ComparatorNetwork, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(parameter_count_report(model), indent=2), encoding="utf-8")
    return target
