"""Shared PyTorch observation encoder, value head, and candidate-scoring policy head."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class TMModelConfig:
    """Architecture and vocabulary configuration for TMNetwork."""

    vocab_size: int
    card_static_dim: int = CARD_STATIC_DIM
    obs_scalar_dim: int = OBS_SCALAR_DIM
    space_feat_dim: int = SPACE_FEAT_DIM
    card_dyn_dim: int = CARD_DYN_DIM
    choice_ctx_dim: int = CHOICE_CTX_DIM
    cand_feat_dim: int = CAND_FEAT_DIM
    num_slots: int = len(SLOT_NAMES)
    num_prompt_types: int = len(PROMPT_TYPES)
    d_model: int = 256
    card_id_emb_dim: int = 128
    card_static_proj_dim: int = 128
    slot_emb_dim: int = 64
    card_dyn_proj_dim: int = 64
    space_hidden_dim: int = 128
    n_heads: int = 8
    n_obs_layers: int = 3
    dim_feedforward: int = 768
    policy_ff_dim: int = 512
    dropout: float = 0.05
    architecture: str = "tm_transformer_v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TMModelConfig:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in payload.items() if k in valid}
        return cls(**filtered)


def _masked_mean_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool `[B, N, D]` over `mask: bool[B, N]`, returning zeros if empty."""
    m = mask.unsqueeze(-1).to(x.dtype)
    denom = m.sum(dim=1).clamp(min=1.0)
    return (x * m).sum(dim=1) / denom


def _masked_max_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Max-pool `[B, N, D]` over `mask: bool[B, N]`, returning zeros if empty."""
    has_any = mask.any(dim=1, keepdim=True)
    x_masked = x.masked_fill(~mask.unsqueeze(-1), -1e4)
    pooled = x_masked.max(dim=1).values
    return torch.where(has_any, pooled, torch.zeros_like(pooled))


class TMNetwork(nn.Module):
    """Shared permutation-invariant observation encoder with value and candidate policy heads."""

    def __init__(self, config: TMModelConfig, card_static_matrix: torch.Tensor) -> None:
        super().__init__()
        if card_static_matrix.shape != (config.vocab_size, config.card_static_dim):
            raise ValueError(
                f"Expected card_static_matrix shape {(config.vocab_size, config.card_static_dim)}, "
                f"got {tuple(card_static_matrix.shape)}"
            )
        self.config = config
        self.register_buffer(
            "card_static_matrix",
            card_static_matrix.detach().clone().to(dtype=torch.float32),
        )

        d = config.d_model
        # Shared card representations (used in both observation tokens and candidate tokens)
        self.card_id_emb = nn.Embedding(
            config.vocab_size, config.card_id_emb_dim, padding_idx=PAD_ID
        )
        self.card_static_proj = nn.Linear(config.card_static_dim, config.card_static_proj_dim)
        self.slot_emb = nn.Embedding(config.num_slots, config.slot_emb_dim, padding_idx=PAD_ID)
        self.card_dyn_proj = nn.Linear(config.card_dyn_dim, config.card_dyn_proj_dim)

        obs_card_in = (
            config.card_id_emb_dim
            + config.card_static_proj_dim
            + config.slot_emb_dim
            + config.card_dyn_proj_dim
        )
        self.card_token_mlp = nn.Sequential(
            nn.Linear(obs_card_in, d),
            nn.LayerNorm(d),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )

        # Public/private scalar encoder -> 1 token
        self.scalar_mlp = nn.Sequential(
            nn.Linear(config.obs_scalar_dim, d),
            nn.LayerNorm(d),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )

        # Permutation-invariant board space set encoder -> 1 summary token (mean + max + attn pool)
        sh = config.space_hidden_dim
        self.space_mlp = nn.Sequential(
            nn.Linear(config.space_feat_dim, sh),
            nn.LayerNorm(sh),
            nn.SiLU(),
            nn.Linear(sh, sh),
            nn.SiLU(),
        )
        self.space_attn_query = nn.Parameter(torch.randn(1, 1, sh) * 0.02)
        self.space_attn = nn.MultiheadAttention(
            embed_dim=sh, num_heads=4, dropout=config.dropout, batch_first=True
        )
        self.space_out_proj = nn.Sequential(
            nn.Linear(sh * 3, d),
            nn.LayerNorm(d),
            nn.SiLU(),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )

        # Observation set transformer (no positional embeddings -> strictly order-invariant)
        obs_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.obs_transformer = nn.TransformerEncoder(
            obs_layer, num_layers=config.n_obs_layers, enable_nested_tensor=False
        )
        self.obs_norm = nn.LayerNorm(d)

        # Multi-head attention pooler for global observation vector
        self.obs_pool_query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.obs_pool_attn = nn.MultiheadAttention(
            embed_dim=d, num_heads=config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.obs_pool_norm = nn.LayerNorm(d)

        # Value head (binary win-probability logit)
        self.value_head = nn.Sequential(
            nn.Linear(d, d),
            nn.LayerNorm(d),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(d, d // 2),
            nn.SiLU(),
            nn.Linear(d // 2, 1),
        )

        # Policy head: choice context + candidate tokens -> cross-attend to obs -> candidate self-attn -> scorer
        self.prompt_emb = nn.Embedding(config.num_prompt_types, 64)
        self.picked_proj = nn.Linear(
            config.card_id_emb_dim + config.card_static_proj_dim, 128
        )
        self.choice_ctx_mlp = nn.Sequential(
            nn.Linear(config.choice_ctx_dim + 64 + 128, d),
            nn.LayerNorm(d),
            nn.SiLU(),
            nn.Linear(d, d),
        )

        cand_in = config.card_id_emb_dim + config.card_static_proj_dim + d // 2
        self.cand_feat_proj = nn.Linear(config.cand_feat_dim, d // 2)
        self.cand_mlp = nn.Sequential(
            nn.Linear(cand_in, d),
            nn.LayerNorm(d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
        self.cand_ctx_gate = nn.Linear(d, d * 2)

        # Cross-attention from candidates to observation tokens
        self.policy_cross_q_norm = nn.LayerNorm(d)
        self.policy_cross_attn = nn.MultiheadAttention(
            embed_dim=d, num_heads=config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.policy_cross_ff_norm = nn.LayerNorm(d)
        self.policy_cross_ff = nn.Sequential(
            nn.Linear(d, config.policy_ff_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.policy_ff_dim, d),
        )

        # 1-layer candidate-to-candidate self-attention (order-invariant across candidate permutations)
        self.policy_self_norm = nn.LayerNorm(d)
        self.policy_self_attn = nn.MultiheadAttention(
            embed_dim=d, num_heads=config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.policy_self_ff_norm = nn.LayerNorm(d)
        self.policy_self_ff = nn.Sequential(
            nn.Linear(d, config.policy_ff_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.policy_ff_dim, d),
        )

        self.policy_scorer = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d // 2),
            nn.SiLU(),
            nn.Linear(d // 2, 1),
        )

    def _embed_cards(self, card_ids: torch.Tensor) -> torch.Tensor:
        id_vec = self.card_id_emb(card_ids)
        static_raw = F.embedding(card_ids, self.card_static_matrix)
        static_vec = self.card_static_proj(static_raw)
        return torch.cat([id_vec, static_vec], dim=-1)

    def encode_observations(
        self, batch: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode observation tensors into `(obs_tokens, obs_mask, obs_pooled)`."""
        obs_scalars = batch["obs_scalars"]
        card_ids = batch["card_ids"]
        card_slots = batch["card_slots"]
        card_dyn = batch["card_dyn"]
        card_mask = batch["card_mask"].bool()
        space_feats = batch["space_feats"]
        space_mask = batch["space_mask"].bool()

        bsz = obs_scalars.shape[0]
        device = obs_scalars.device

        # 1. Scalar token [B, 1, D]
        scalar_tok = self.scalar_mlp(obs_scalars).unsqueeze(1)

        # 2. Space set summary token [B, 1, D]
        space_h = self.space_mlp(space_feats)
        sp_mean = _masked_mean_pool(space_h, space_mask)
        sp_max = _masked_max_pool(space_h, space_mask)
        has_sp = space_mask.any(dim=1)
        safe_sp_mask = space_mask.clone()
        safe_sp_mask[~has_sp, 0] = True
        sp_q = self.space_attn_query.expand(bsz, -1, -1)
        sp_attn, _ = self.space_attn(
            sp_q, space_h, space_h, key_padding_mask=~safe_sp_mask, need_weights=False
        )
        sp_attn = sp_attn.squeeze(1) * has_sp.unsqueeze(-1).to(space_h.dtype)
        space_tok = self.space_out_proj(torch.cat([sp_mean, sp_max, sp_attn], dim=-1)).unsqueeze(1)

        # 3. Card tokens [B, N_cards, D]
        card_base = self._embed_cards(card_ids)
        slot_vec = self.slot_emb(card_slots)
        dyn_vec = self.card_dyn_proj(card_dyn)
        card_toks = self.card_token_mlp(torch.cat([card_base, slot_vec, dyn_vec], dim=-1))
        card_toks = card_toks * card_mask.unsqueeze(-1).to(card_toks.dtype)

        # Concatenate all observation tokens; first 2 tokens are always valid
        obs_tokens = torch.cat([scalar_tok, space_tok, card_toks], dim=1)
        prefix_mask = torch.ones((bsz, 2), dtype=torch.bool, device=device)
        obs_mask = torch.cat([prefix_mask, card_mask], dim=1)

        obs_tokens = self.obs_transformer(obs_tokens, src_key_padding_mask=~obs_mask)
        obs_tokens = self.obs_norm(obs_tokens)

        pool_q = self.obs_pool_query.expand(bsz, -1, -1)
        pooled, _ = self.obs_pool_attn(
            pool_q, obs_tokens, obs_tokens, key_padding_mask=~obs_mask, need_weights=False
        )
        obs_pooled = self.obs_pool_norm(pooled.squeeze(1) + scalar_tok.squeeze(1))
        return obs_tokens, obs_mask, obs_pooled

    def forward_value(self, obs_pooled: torch.Tensor) -> torch.Tensor:
        """Return win-probability logit `[B]` from pooled observation embedding."""
        return self.value_head(obs_pooled).squeeze(-1)

    def forward_policy(
        self,
        obs_tokens: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_pooled: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score candidates conditioned on encoded observation tokens.

        Returns `(masked_logits [B, C], probs [B, C])` where masked positions have
        `-inf` logits and exact `0.0` probability.
        """
        choice_ctx = batch["choice_ctx"]
        prompt_type_ids = batch["prompt_type_ids"]
        cand_card_ids = batch["cand_card_ids"]
        cand_feats = batch["cand_feats"]
        cand_mask = batch["cand_mask"].bool()
        picked_card_ids = batch["picked_card_ids"]
        picked_mask = batch["picked_mask"].bool()

        # Choice context vector [B, D]
        p_emb = self.prompt_emb(prompt_type_ids)
        picked_base = self._embed_cards(picked_card_ids)
        picked_vec = self.picked_proj(_masked_mean_pool(picked_base, picked_mask))
        ctx_vec = self.choice_ctx_mlp(torch.cat([choice_ctx, p_emb, picked_vec], dim=-1))

        # Candidate initial tokens [B, C, D]
        cand_base = self._embed_cards(cand_card_ids)
        cand_f = self.cand_feat_proj(cand_feats)
        c_tok = self.cand_mlp(torch.cat([cand_base, cand_f], dim=-1))

        # Condition candidates on global obs summary + choice context via FiLM
        gamma, beta = self.cand_ctx_gate(ctx_vec + obs_pooled).chunk(2, dim=-1)
        c_tok = c_tok * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        # Cross-attention from candidates -> observation tokens
        q = self.policy_cross_q_norm(c_tok)
        cross_out, _ = self.policy_cross_attn(
            q, obs_tokens, obs_tokens, key_padding_mask=~obs_mask, need_weights=False
        )
        c_tok = c_tok + cross_out
        c_tok = c_tok + self.policy_cross_ff(self.policy_cross_ff_norm(c_tok))

        # Candidate self-attention (guard against degenerate all-masked rows)
        has_cand = cand_mask.any(dim=1)
        safe_cand_mask = cand_mask.clone()
        safe_cand_mask[~has_cand, 0] = True
        sq = self.policy_self_norm(c_tok)
        self_out, _ = self.policy_self_attn(
            sq, sq, sq, key_padding_mask=~safe_cand_mask, need_weights=False
        )
        c_tok = c_tok + self_out
        c_tok = c_tok + self.policy_self_ff(self.policy_self_ff_norm(c_tok))

        raw_logits = self.policy_scorer(c_tok).squeeze(-1).float()
        masked_logits = raw_logits.masked_fill(~cand_mask, float("-inf"))

        probs = torch.zeros_like(masked_logits)
        if has_cand.any():
            valid_probs = F.softmax(masked_logits[has_cand], dim=-1)
            probs[has_cand] = valid_probs
        probs = torch.where(cand_mask, probs, torch.zeros_like(probs))
        return masked_logits, probs

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        compute_policy: bool = True,
        compute_value: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Forward pass supporting policy-only, value-only, or joint evaluation."""
        obs_tokens, obs_mask, obs_pooled = self.encode_observations(batch)
        out: dict[str, torch.Tensor] = {"obs_pooled": obs_pooled}
        if compute_value:
            value_logit = self.forward_value(obs_pooled).float()
            win_prob = torch.sigmoid(value_logit)
            out["value_logit"] = value_logit
            out["win_prob"] = win_prob
            out["value"] = 2.0 * win_prob - 1.0
        if compute_policy and "cand_card_ids" in batch:
            logits, probs = self.forward_policy(obs_tokens, obs_mask, obs_pooled, batch)
            out["policy_logits"] = logits
            out["policy_probs"] = probs
        return out

    def count_parameters(self, *, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_vocab(
        cls,
        vocab: CardVocab,
        config: TMModelConfig | None = None,
    ) -> TMNetwork:
        cfg = config if config is not None else TMModelConfig(vocab_size=vocab.size)
        if cfg.vocab_size != vocab.size:
            raise ValueError(
                f"TMModelConfig.vocab_size ({cfg.vocab_size}) != vocab.size ({vocab.size})"
            )
        static_mat = torch.from_numpy(vocab.static_features)
        return cls(config=cfg, card_static_matrix=static_mat)


def save_checkpoint(
    path: str | Path,
    model: TMNetwork,
    vocab: CardVocab,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save model weights, architecture config, vocab, and parameter count.

    If `path` points to a directory, writes `checkpoint.pt` and `config.json` inside it.
    If `path` ends with `.pt`/`.pth`, writes that file plus a sibling `config.json`.
    """
    target = Path(path)
    if target.suffix in (".pt", ".pth", ".ckpt"):
        ckpt_path = target
        out_dir = target.parent
    else:
        out_dir = target
        ckpt_path = out_dir / "checkpoint.pt"
    out_dir.mkdir(parents=True, exist_ok=True)

    param_count = model.count_parameters(trainable_only=False)
    trainable_count = model.count_parameters(trainable_only=True)
    cfg_dict = model.config.to_dict()
    vocab_dict = vocab.to_dict()

    payload = {
        "model_state_dict": model.state_dict(),
        "config": cfg_dict,
        "vocab": vocab_dict,
        "parameter_count": param_count,
        "trainable_parameter_count": trainable_count,
        "metadata": dict(metadata) if metadata is not None else {},
    }
    torch.save(payload, ckpt_path)

    config_json = {
        "config": cfg_dict,
        "parameter_count": param_count,
        "trainable_parameter_count": trainable_count,
        "vocab_size": vocab.size,
        "vocab_names": list(vocab.names),
        "metadata": dict(metadata) if metadata is not None else {},
    }
    (out_dir / "config.json").write_text(
        json.dumps(config_json, indent=2), encoding="utf-8"
    )
    return ckpt_path


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[TMNetwork, CardVocab, dict[str, Any]]:
    """Load `(model, vocab, metadata)` saved by `save_checkpoint`."""
    target = Path(path)
    ckpt_path = target / "checkpoint.pt" if target.is_dir() else target
    payload = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    cfg = TMModelConfig.from_dict(payload["config"])
    vocab = CardVocab.from_dict(payload["vocab"])
    model = TMNetwork.from_vocab(vocab, cfg)
    model.load_state_dict(payload["model_state_dict"])
    model = model.to(map_location)
    model.eval()
    metadata = dict(payload.get("metadata", {}))
    return model, vocab, metadata


def build_random_init_model(
    path: str | Path,
    *,
    seed: int,
    map_location: str | torch.device = "cpu",
) -> tuple[TMNetwork, CardVocab, dict[str, Any]]:
    """Build a fresh model from a checkpoint's architecture and vocabulary (#27).

    The checkpoint supplies only the `TMModelConfig`, the `CardVocab`, and the
    non-learned `card_static_matrix` buffer. Every learned parameter is drawn
    from `seed`, so two runs with the same seed start from identical weights
    regardless of the checkpoint's weights or the caller's global RNG state.
    `load_checkpoint` sets no other non-learned attributes (`TMNetwork` has no
    `temperature` attribute, and `config`/`card_static_matrix` are the only
    non-learned state), so nothing else is carried over.
    """
    warm, vocab, metadata = load_checkpoint(path, map_location="cpu")
    # Parameters are drawn on CPU, so fork only the CPU RNG and leave the
    # caller's (and any CUDA) random state untouched.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = TMNetwork(warm.config, warm.card_static_matrix)
    model = model.to(map_location)
    model.eval()
    return model, vocab, metadata
