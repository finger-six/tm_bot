"""Comparator self-play: direct-policy games, opponent sampling, and PPO rows."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from tm_bot.agents import B1Agent, load_card_data
from tm_bot.agents.protocol import Agent, GameRunResult
from tm_bot.agents.runner import play_games
from tm_bot.comparator.agent import ComparatorAgent
from tm_bot.comparator.model import ComparatorNetwork, load_comparator_checkpoint
from tm_bot.comparator.ppo import PPOSample, compute_gae
from tm_bot.models.encode import collate_encoded_batch, encode_choice, encode_observation
from tm_bot.models.vocab import CardVocab
from tm_bot.runs.buffer import normalize_policy_target
from tm_bot.runs.config import RunConfig
from tm_bot.runs.recorder import GameRecorder, capture_agent
from tm_bot.runs.selfplay import (
    OpponentChoice,
    OpponentSampler,
    SelfPlayBatch,
    build_selfplay_specs,
    game_summary,
    make_train_seed_keys,
    utility_for_seat,
)


class ComparatorSelfPlayEngine:
    """Plays self-play games with the direct comparator policy and sampled opponents."""

    def __init__(
        self,
        config: RunConfig,
        *,
        model: ComparatorNetwork,
        vocab: CardVocab,
        device: str = "cpu",
        rng: np.random.Generator | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.vocab = vocab
        self.device = device
        self.rng = rng or np.random.default_rng(config.seed)
        self.opponents = OpponentSampler(
            latest_prob=config.opponent_latest_prob,
            pool_prob=config.opponent_pool_prob,
            b1_prob=config.opponent_b1_prob,
            rng=self.rng,
        )
        self._card_data = None
        self._pool_agents: dict[str, ComparatorAgent] = {}
        self.seed_cursor = config.train_seed_start

    def set_opponent_pool(self, checkpoints: Sequence[str | Path]) -> None:
        self.opponents.set_pool(checkpoints)
        keep = {str(Path(p)) for p in checkpoints}
        for key in list(self._pool_agents):
            if key not in keep:
                del self._pool_agents[key]

    def _learner_factory(self, seed: int) -> ComparatorAgent:
        return ComparatorAgent(
            self.model,
            self.vocab,
            device=self.device,
            temperature=self.config.action_temperature,
            seed=seed,
            version="comparator-latest",
        )

    def _latest_opponent_factory(self, seed: int) -> ComparatorAgent:
        return ComparatorAgent(
            self.model,
            self.vocab,
            device=self.device,
            temperature=0.0,
            seed=seed,
            version="comparator-latest-greedy",
        )

    def _pool_factory(self, path: Path, seed: int) -> ComparatorAgent:
        key = str(path)
        cached = self._pool_agents.get(key)
        if cached is None:
            model, vocab, _meta = load_comparator_checkpoint(path, map_location=self.device)
            cached = ComparatorAgent(
                model,
                vocab,
                device=self.device,
                temperature=0.0,
                seed=seed,
                version=f"comparator-{path.name}",
            )
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
        return self._latest_opponent_factory(seed)

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
    ) -> tuple[np.ndarray, np.ndarray]:
        """Recompute behavior log-probs and raw win probabilities for recorded decisions."""
        if not records:
            return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
        enc_obs = [encode_observation(r.observation, self.vocab) for r in records]
        enc_choice = [
            encode_choice(r.choice, r.observation, self.vocab) for r in records
        ]
        batch = collate_encoded_batch(enc_obs, enc_choice)
        tensors = {k: v.to(self.device) for k, v in batch.items()}
        self.model.eval()
        out = self.model(tensors, compute_policy=True, compute_value=True)
        log_probs = torch.log_softmax(out["policy_logits"].float(), dim=-1)
        actions = torch.tensor(
            [r.candidate_index for r in records], dtype=torch.long, device=log_probs.device
        )
        chosen = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        values = out["win_prob"].float()
        return chosen.detach().cpu().numpy(), values.detach().cpu().numpy()

    def assemble_batch(
        self,
        outcomes: Sequence[GameRunResult],
        recorders: Sequence[GameRecorder],
        training_seats: Sequence[int],
        opponent_kinds: Sequence[str] | None = None,
    ) -> SelfPlayBatch:
        """Encode recorded training-seat decisions into PPO rows with GAE targets."""
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
            utility = utility_for_seat(summary, seat)
            records = [r for r in recorder.decisions if int(r.observation["actor"]) == seat]
            if len(records) != len(recorder.decisions):
                raise RuntimeError(
                    "recorded comparator observations do not all belong to the training seat"
                )
            old_log_probs, behavior_values = self._behavior_stats(records)
            advantages, returns = compute_gae(
                behavior_values,
                utility,
                gamma=self.config.ppo_gamma,
                gae_lambda=self.config.ppo_gae_lambda,
            )
            for rec, old_logp, value, advantage, ret in zip(
                records, old_log_probs, behavior_values, advantages, returns
            ):
                n = len(rec.choice.candidates)
                if n < 1:
                    continue
                ppo_samples.append(
                    PPOSample(
                        observation=encode_observation(rec.observation, self.vocab),
                        choice=encode_choice(rec.choice, rec.observation, self.vocab),
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
                    )
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
