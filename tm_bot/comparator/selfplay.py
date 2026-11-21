"""Comparator self-play: direct-policy games, opponent sampling, and PPO rows."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from tm_bot.agents import B1Agent, load_card_data
from tm_bot.agents.play_rules import game_utility
from tm_bot.agents.protocol import Agent, GameRunResult
from tm_bot.agents.runner import play_games
from tm_bot.comparator.agent import ComparatorAgent
from tm_bot.comparator.model import ComparatorNetwork, load_comparator_checkpoint
from tm_bot.comparator.ppo import (
    PPOSample,
    compute_gae,
    critic_baseline,
    sampling_temperatures,
    scaled_log_probs,
)
from tm_bot.models.encode import collate_encoded_batch
from tm_bot.models.vocab import CardVocab
from tm_bot.runs.anchor import anchor_policy_logits, check_anchor_compatible, freeze
from tm_bot.runs.buffer import normalize_policy_target
from tm_bot.runs.config import RunConfig
from tm_bot.runs.recorder import DecisionRecord, GameRecorder
from tm_bot.runs.selfplay import (
    OpponentChoice,
    OpponentSampler,
    SelfPlayBatch,
    build_selfplay_specs,
    check_recorded_seats,
    game_summary,
    make_train_seed_keys,
    margin_for_seat,
    utility_for_seat,
)


class ComparatorSelfPlayEngine:
    """Plays self-play games with the direct comparator policy and sampled opponents.

    Every learned seat (learner, `latest`, pool, warm-start anchor) samples at τ(g) in
    the action phase and τ 1 elsewhere (#39, #48); there is no greedy seat.
    """

    def __init__(
        self,
        config: RunConfig,
        *,
        model: ComparatorNetwork,
        vocab: CardVocab,
        device: str = "cpu",
        rng: np.random.Generator | None = None,
        anchor_model: ComparatorNetwork | None = None,
        anchor_vocab: CardVocab | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.vocab = vocab
        self.device = device
        self.rng = rng or np.random.default_rng(config.seed)
        self.opponents = OpponentSampler(
            latest_prob=config.opponent_latest_prob,
            pool_prob=config.opponent_pool_prob,
            anchor_prob=config.opponent_anchor_prob,
            b1_prob=config.opponent_b1_prob,
            rng=self.rng,
        )
        self._card_data = None
        self._pool_agents: dict[str, ComparatorAgent] = {}
        self._anchor: tuple[ComparatorNetwork, CardVocab] | None = None
        if anchor_model is not None:
            self._set_anchor(anchor_model, anchor_vocab or vocab)
        self.seed_cursor = config.train_seed_start

    def _set_anchor(self, model: ComparatorNetwork, vocab: CardVocab) -> None:
        check_anchor_compatible(model, vocab, self.model, self.vocab)
        freeze(model)
        self._anchor = (model, vocab)

    @property
    def anchor(self) -> tuple[ComparatorNetwork, CardVocab] | None:
        """The frozen warm-start anchor `(model, vocab)`, loaded once; None without one."""
        if self._anchor is None and self.config.warmstart_checkpoint:
            model, vocab, _meta = load_comparator_checkpoint(
                self.config.warmstart_checkpoint, map_location=self.device
            )
            self._set_anchor(model, vocab)
        return self._anchor

    def _agent(
        self, model: ComparatorNetwork, vocab: CardVocab, seed: int, version: str
    ) -> ComparatorAgent:
        return ComparatorAgent(
            model,
            vocab,
            device=self.device,
            temperature=1.0,
            play_temperature=self.config.play_temperature,
            seed=seed,
            version=version,
        )

    def set_opponent_pool(self, checkpoints: Sequence[str | Path]) -> None:
        self.opponents.set_pool(checkpoints)
        keep = {str(Path(p)) for p in checkpoints}
        for key in list(self._pool_agents):
            if key not in keep:
                del self._pool_agents[key]

    def _learner_factory(self, seed: int) -> ComparatorAgent:
        return self._agent(self.model, self.vocab, seed, "comparator-latest")

    def _anchor_factory(self, seed: int) -> ComparatorAgent:
        anchor = self.anchor
        if anchor is None:
            raise RuntimeError(
                "the anchor opponent needs config.warmstart_checkpoint (the frozen warm start)"
            )
        return self._agent(anchor[0], anchor[1], seed, "comparator-anchor")

    def _pool_factory(self, path: Path, seed: int) -> ComparatorAgent:
        key = str(path)
        cached = self._pool_agents.get(key)
        if cached is None:
            model, vocab, _meta = load_comparator_checkpoint(
                path, map_location=self.device, encoder_version=None
            )
            cached = self._agent(model, vocab, seed, f"comparator-{path.name}")
            self._pool_agents[key] = cached
        agent = copy.deepcopy(cached)
        agent.reset(seed)
        return agent

    def _opponent_factory(self, choice: OpponentChoice, seed: int) -> Agent:
        if choice.kind == "b1":
            if self._card_data is None:
                self._card_data = load_card_data()
            return B1Agent(seed=seed, card_data=self._card_data)
        if choice.kind == "pool" and choice.checkpoint is not None:
            return self._pool_factory(choice.checkpoint, seed)
        if choice.kind == "anchor":
            return self._anchor_factory(seed)
        return self._learner_factory(seed)

    def play_batch(
        self,
        num_games: int,
        *,
        stop_when: Any | None = None,
    ) -> SelfPlayBatch:
        if num_games < 1:
            raise ValueError("num_games must be >= 1")
        keys = make_train_seed_keys(self.seed_cursor, num_games)
        self.seed_cursor += num_games
        specs, recorders, training_seats, kinds = build_selfplay_specs(
            seed_keys=keys,
            maps=self.config.maps,
            learner_factory=self._learner_factory,
            sampler=self.opponents,
            opponent_factory=self._opponent_factory,
            vocab=self.vocab,
            seed_start=self.seed_cursor,
        )
        outcomes = play_games(
            specs,
            num_workers=self.config.workers,
            batch_size=self.config.batch_size,
            max_decisions=self.config.max_decisions_per_game,
            stop_when=stop_when,
        )
        return self.assemble_batch(outcomes, recorders, training_seats, kinds)

    @torch.inference_mode()
    def _behavior_stats(
        self, records: Sequence[Any]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Recompute behavior stats from stored encodings: `(log_probs, win_probs, baselines)`.

        `log_probs` are under the τ-scaled policy each decision was sampled from (its
        `play_temperature`), the same `scaled_log_probs` the learner uses. `baselines` are
        the critic (2p − 1) + 0.15·m̂ in U units; `win_probs` are the raw p.
        """
        if not records:
            empty = np.zeros((0,), dtype=np.float64)
            return empty, empty, empty
        batch = collate_encoded_batch(
            [r.observation for r in records],
            [r.choice for r in records],
        )
        tensors = {k: v.to(self.device) for k, v in batch.items()}
        self.model.eval()
        out = self.model(tensors, compute_policy=True, compute_value=True)
        log_probs = scaled_log_probs(
            out["policy_logits"],
            sampling_temperatures([r.play_temperature for r in records]),
            tensors["cand_mask"],
        )
        actions = torch.tensor(
            [r.candidate_index for r in records], dtype=torch.long, device=log_probs.device
        )
        chosen = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        win_prob = out["win_prob"].float().detach().cpu().numpy()
        margin = out.get("vp_margin_pred")
        baselines = critic_baseline(
            win_prob, None if margin is None else margin.float().detach().cpu().numpy()
        )
        return chosen.detach().cpu().numpy(), win_prob, baselines

    def assemble_batch(
        self,
        outcomes: Sequence[GameRunResult],
        recorders: Sequence[GameRecorder],
        training_seats: Sequence[int],
        opponent_kinds: Sequence[str] | None = None,
    ) -> SelfPlayBatch:
        """Encode recorded decisions into PPO rows with GAE targets.

        Mirror games carry both seats; each seat is its own trajectory, with GAE run on
        that seat's U = o + 0.15·m. `old_log_prob` is the log-probability under the
        τ-scaled policy the decision was sampled from (`play_temperature`).
        """
        games = [game_summary(o) for o in outcomes]
        ppo_samples: list[PPOSample] = []
        latencies: list[float] = []
        decisions = 0
        for outcome, recorder, seat in zip(outcomes, recorders, training_seats):
            decisions += int(outcome.decisions)
            latencies.extend(outcome.decision_latencies_ms[0])
            latencies.extend(outcome.decision_latencies_ms[1])
            summary = game_summary(outcome)
            if not summary.ok:
                continue
            if not recorder.decisions:
                continue
            check_recorded_seats(recorder, seat)
            for traj_seat in (seat, 1 - seat):
                records = [r for r in recorder.decisions if int(r.seat) == traj_seat]
                if records:
                    ppo_samples.extend(
                        self._trajectory_samples(recorder, summary, traj_seat, records)
                    )
        return SelfPlayBatch(
            games=games,
            samples=[],
            recorders=list(recorders),
            training_seats=list(training_seats),
            opponent_kinds=list(opponent_kinds or []),
            decisions=decisions,
            searched_decisions=0,
            latencies_ms=latencies,
            seed_keys=[o.seed_key for o in outcomes],
            ppo_samples=ppo_samples,
        )

    def _trajectory_samples(
        self,
        recorder: GameRecorder,
        summary: Any,
        seat: int,
        records: Sequence[DecisionRecord],
    ) -> list[PPOSample]:
        utility = utility_for_seat(summary, seat)
        margin = margin_for_seat(summary, seat)
        old_log_probs, behavior_values, baselines = self._behavior_stats(records)
        anchor_logits = self._anchor_logits(records)
        # Reward U = o + 0.15·m against the critic baseline (2p − 1) + 0.15·m̂, both in U
        # units. The win head's own target stays the outcome o (`value_target`).
        advantages, returns = compute_gae(
            baselines,
            game_utility(utility, summary.scores[seat], summary.scores[1 - seat]),
            gamma=self.config.ppo_gamma,
            gae_lambda=self.config.ppo_gae_lambda,
        )
        samples: list[PPOSample] = []
        for rec, old_logp, value, advantage, ret, logits in zip(
            records, old_log_probs, behavior_values, advantages, returns, anchor_logits
        ):
            n = rec.num_candidates
            if n < 1:
                continue
            samples.append(
                PPOSample(
                    observation=rec.observation,
                    choice=rec.choice,
                    policy_target=normalize_policy_target(rec.policy, n),
                    value_target=float(utility),
                    seat=int(seat),
                    was_searched=False,
                    seed_key=recorder.seed_key,
                    map_name=recorder.map_name,
                    generation=int(rec.generation),
                    action=int(rec.candidate_index),
                    old_log_prob=float(old_logp),
                    behavior_value=float(value),
                    advantage=float(advantage),
                    return_=float(ret),
                    play_temperature=rec.play_temperature,
                    margin_target=margin,
                    anchor_logits=logits,
                )
            )
        return samples

    def _anchor_logits(self, records: Sequence[DecisionRecord]) -> list[np.ndarray | None]:
        anchor = self.anchor
        if anchor is None:
            return [None] * len(records)
        return list(
            anchor_policy_logits(
                anchor[0],
                [r.observation for r in records],
                [r.choice for r in records],
                [r.num_candidates for r in records],
                device=self.device,
            )
        )
