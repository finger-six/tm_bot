"""Self-play game generation on `train-*` seeds and opponent sampling."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tm_bot.agents import B1Agent, GameSpec, load_card_data
from tm_bot.agents.protocol import Agent, GameRunResult
from tm_bot.agents.runner import DEFAULT_MAX_DECISIONS, play_games
from tm_bot.env.client import WorkerClient
from tm_bot.eval.seeds import (
    SeedKey,
    derive_agent_seed,
    generate_seed_keys,
    make_seed_key,
    validate_namespace,
)
from tm_bot.models.encode import encode_choice, encode_observation
from tm_bot.models.network import TMNetwork
from tm_bot.models.vocab import CardVocab
from tm_bot.runs.buffer import DecisionSample, normalize_policy_target
from tm_bot.runs.config import RunConfig
from tm_bot.runs.monitoring import (
    GameSummary,
    DeterminismResult,
    compare_replay,
    policy_entropy,
)
from tm_bot.runs.recorder import GameRecorder, capture_agent
from tm_bot.search.agent import ISMCTSAgent


@dataclass(frozen=True, slots=True)
class OpponentChoice:
    """One sampled opponent: the latest net, a frozen pool checkpoint, or B1."""

    kind: str  # "latest" | "pool" | "b1"
    checkpoint: Path | None = None


class OpponentSampler:
    """Sample opponents 60% latest / 30% last-10 pool / 10% B1 by default."""

    def __init__(
        self,
        *,
        latest_prob: float = 0.60,
        pool_prob: float = 0.30,
        b1_prob: float = 0.10,
        rng: np.random.Generator | None = None,
    ) -> None:
        if abs(latest_prob + pool_prob + b1_prob - 1.0) > 1e-9:
            raise ValueError("opponent probabilities must sum to 1")
        self.latest_prob = latest_prob
        self.pool_prob = pool_prob
        self.b1_prob = b1_prob
        self.rng = rng or np.random.default_rng(0)
        self.pool: tuple[Path, ...] = ()

    def set_pool(self, checkpoints: Sequence[str | Path]) -> None:
        self.pool = tuple(Path(p) for p in checkpoints)

    def sample(self) -> OpponentChoice:
        roll = float(self.rng.random())
        if self.pool and roll >= self.latest_prob and roll < self.latest_prob + self.pool_prob:
            idx = int(self.rng.integers(0, len(self.pool)))
            return OpponentChoice(kind="pool", checkpoint=self.pool[idx])
        if roll >= self.latest_prob + self.pool_prob and self.b1_prob > 0.0:
            return OpponentChoice(kind="b1")
        return OpponentChoice(kind="latest")


def make_train_seed_keys(start: int, count: int) -> tuple[SeedKey, ...]:
    """Consecutive `train-*` seed keys; never dev or eval."""
    keys = generate_seed_keys("train", count, start=start)
    assert_training_seed_keys([k.key for k in keys])
    return keys


def assert_training_seed_keys(seed_keys: Sequence[str]) -> None:
    """Reject any seed key that is not in the `train-*` namespace."""
    for raw in seed_keys:
        parsed = SeedKey.parse(str(raw))
        if parsed.namespace != "train":
            raise ValueError(
                f"Self-play seed {raw!r} is in the {parsed.namespace!r} namespace; "
                "training may only use train-* seeds"
            )


def assert_namespace(seed_keys: Sequence[str], namespace: str) -> None:
    """Reject any seed key outside `namespace` (used for dev/eval guards)."""
    expected = validate_namespace(namespace)
    for raw in seed_keys:
        parsed = SeedKey.parse(str(raw))
        if parsed.namespace != expected:
            raise ValueError(
                f"Seed {raw!r} is in the {parsed.namespace!r} namespace; expected {expected!r}"
            )


def build_selfplay_specs(
    *,
    seed_keys: Sequence[SeedKey],
    maps: Sequence[str],
    learner_factory: Any,
    sampler: OpponentSampler,
    opponent_factory: Any,
    seed_start: int = 0,
) -> tuple[list[GameSpec], list[GameRecorder], list[int], list[str]]:
    """Build one-game-per-seed specs with a training seat and a sampled opponent.

    `learner_factory(seed)` and `opponent_factory(choice, seed)` must return
    `Agent`s; the recorder's `capture_seat` is the training seat, so only that
    seat's `(observation, choice)` pairs become training rows. Returns
    `(specs, recorders, training_seats, opponent_kinds)`.
    """
    if not maps:
        raise ValueError("maps must be non-empty")
    assert_training_seed_keys([k.key for k in seed_keys])

    specs: list[GameSpec] = []
    recorders: list[GameRecorder] = []
    training_seats: list[int] = []
    opponent_kinds: list[str] = []
    for i, seed_key in enumerate(seed_keys):
        map_name = maps[(seed_start + i) % len(maps)]
        training_seat = i % 2
        opponent = sampler.sample()
        learner_seed = derive_agent_seed(seed_key, map_name=map_name, role="train")
        opponent_seed = derive_agent_seed(seed_key, map_name=map_name, role="opp")
        learner = learner_factory(learner_seed)
        opp_agent = opponent_factory(opponent, opponent_seed)

        recorder = GameRecorder(
            seed_key=seed_key.key,
            map_name=map_name,
            capture_seat=training_seat,  # type: ignore[arg-type]
        )
        captured_learner = capture_agent(learner, recorder, capture_seat=training_seat)  # type: ignore[arg-type]
        captured_opp = capture_agent(opp_agent, recorder, capture_seat=None)
        agents: tuple[Agent, Agent] = (
            (captured_learner, captured_opp)
            if training_seat == 0
            else (captured_opp, captured_learner)
        )
        seat_seeds = (
            (learner_seed, opponent_seed)
            if training_seat == 0
            else (opponent_seed, learner_seed)
        )
        specs.append(
            GameSpec(
                seed_key=seed_key.key,
                map_name=map_name,
                agents=agents,
                seat_seeds=seat_seeds,
            )
        )
        recorders.append(recorder)
        training_seats.append(training_seat)
        opponent_kinds.append(opponent.kind)
    return specs, recorders, training_seats, opponent_kinds


def game_summary(outcome: GameRunResult) -> GameSummary:
    res = outcome.result
    return GameSummary(
        seed_key=outcome.seed_key,
        map_name=outcome.map,
        ok=bool(outcome.ok and res is not None),
        generations=outcome.generations,
        decisions=int(outcome.decisions),
        scores=None if res is None else (int(res.scores[0]), int(res.scores[1])),
        winner=None if res is None else (None if res.winner is None else int(res.winner)),
        tie=False if res is None else bool(res.tie),
        error_type=outcome.error_type,
        error=outcome.error,
    )


def utility_for_seat(summary: GameSummary, seat: int) -> float:
    """+1 win, -1 loss, 0 tie from `seat`'s perspective."""
    if not summary.ok or summary.scores is None:
        raise ValueError("cannot score an unfinished game")
    if summary.tie:
        return 0.0
    if summary.winner is None:
        raise ValueError(
            f"game {summary.seed_key} has no declared winner and is not a tie"
        )
    return 1.0 if summary.winner == seat else -1.0


@dataclass(slots=True)
class SelfPlayBatch:
    """Games played in one self-play cycle plus their encoded training rows."""

    games: list[GameSummary]
    samples: list[DecisionSample]
    recorders: list[GameRecorder] = field(default_factory=list)
    training_seats: list[int] = field(default_factory=list)
    opponent_kinds: list[str] = field(default_factory=list)
    decisions: int = 0
    searched_decisions: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    seed_keys: list[str] = field(default_factory=list)
    # Comparator runs fill PPO rollouts instead of AlphaZero decision samples.
    ppo_samples: list[Any] | None = None

    @property
    def training_samples(self) -> list[Any]:
        """Rows the learner trains on (PPO rollouts for the comparator, else AlphaZero rows)."""
        if self.ppo_samples is not None:
            return list(self.ppo_samples)
        return list(self.samples)

    @property
    def entropy(self) -> float | None:
        items = self.training_samples
        if not items:
            return None
        return float(np.mean([policy_entropy(s.policy_target) for s in items]))

    @property
    def failures(self) -> list[GameSummary]:
        return [g for g in self.games if not g.ok]


class SelfPlayEngine:
    """Plays self-play games with the in-memory latest network and sampled opponents."""

    def __init__(
        self,
        config: RunConfig,
        *,
        model: TMNetwork,
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
        self._pool_agents: dict[str, ISMCTSAgent] = {}
        self.seed_cursor = config.train_seed_start

    def set_opponent_pool(self, checkpoints: Sequence[str | Path]) -> None:
        self.opponents.set_pool(checkpoints)
        keep = {str(Path(p)) for p in checkpoints}
        for key in list(self._pool_agents):
            if key not in keep:
                del self._pool_agents[key]

    def _learner_factory(self, seed: int) -> ISMCTSAgent:
        from tm_bot.models.agent import PolicyValueAgent

        pv = PolicyValueAgent(
            self.model,
            self.vocab,
            device=self.device,
            temperature=0.0,
            seed=seed,
            version="selfplay-latest",
        )
        return ISMCTSAgent(pv, config=self.config.search_config(), seed=seed)

    def _pool_factory(self, path: Path, seed: int) -> ISMCTSAgent:
        key = str(path)
        cached = self._pool_agents.get(key)
        if cached is None:
            cached = ISMCTSAgent.from_checkpoint(
                path,
                device=self.device,
                config=self.config.search_config(),
                policy_temperature=0.0,
                seed=seed,
            )
            self._pool_agents[key] = cached
        import copy

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
        return self._learner_factory(seed)

    def play_batch(
        self,
        num_games: int,
        *,
        stop_when: Any | None = None,
    ) -> SelfPlayBatch:
        """Play `num_games` self-play games on fresh `train-*` seeds.

        `stop_when` is polled between runner steps; when it returns True the
        in-flight batch is cut short and the partial games are reported with
        `error_type="DeadlineReached"` (the orchestrator treats that as a budget
        stop, not an engine failure).
        """
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

    def assemble_batch(
        self,
        outcomes: Sequence[GameRunResult],
        recorders: Sequence[GameRecorder],
        training_seats: Sequence[int],
        opponent_kinds: Sequence[str] | None = None,
    ) -> SelfPlayBatch:
        """Encode recorded acting-seat decisions into training rows with utility targets."""
        games = [game_summary(o) for o in outcomes]
        samples: list[DecisionSample] = []
        latencies: list[float] = []
        decisions = 0
        searched = 0
        for outcome, recorder, seat in zip(outcomes, recorders, training_seats):
            decisions += int(outcome.decisions)
            searched += recorder.num_searched
            latencies.extend(outcome.decision_latencies_ms[0])
            latencies.extend(outcome.decision_latencies_ms[1])
            summary = game_summary(outcome)
            if not summary.ok:
                continue
            value_target = utility_for_seat(summary, seat)
            for rec in recorder.decisions:
                if int(rec.observation["actor"]) != seat:
                    raise RuntimeError(
                        f"recorded observation actor {rec.observation['actor']} "
                        f"does not match training seat {seat}"
                    )
                n = len(rec.choice.candidates)
                if n < 1:
                    continue
                samples.append(
                    DecisionSample(
                        observation=encode_observation(rec.observation, self.vocab),
                        choice=encode_choice(rec.choice, rec.observation, self.vocab),
                        policy_target=normalize_policy_target(rec.policy, n),
                        value_target=float(value_target),
                        seat=int(seat),
                        was_searched=bool(rec.was_searched),
                        seed_key=recorder.seed_key,
                        map_name=recorder.map_name,
                        generation=int(rec.generation),
                    )
                )
        return SelfPlayBatch(
            games=games,
            samples=samples,
            recorders=list(recorders),
            training_seats=list(training_seats),
            opponent_kinds=list(opponent_kinds or []),
            decisions=decisions,
            searched_decisions=searched,
            latencies_ms=latencies,
            seed_keys=[o.seed_key for o in outcomes],
        )


def replay_game(
    seed_key: str,
    map_name: str,
    actions: Sequence[tuple[int, int]],
    *,
    max_decisions: int = DEFAULT_MAX_DECISIONS,
) -> GameSummary:
    """Replay a finished game from its seed and recorded (seat, candidate) actions."""
    decisions = 0
    with WorkerClient() as active:
        game = active.open(seed_key=seed_key, map=map_name)
        try:
            for seat, index in actions:
                pending = game.pending()
                if seat not in pending:
                    raise ValueError(
                        f"Replay action {decisions} for seat {seat} but pending seats are {pending}"
                    )
                game.choose(seat, int(index))  # type: ignore[arg-type]
                decisions += 1
                if decisions > max_decisions:
                    raise ValueError(f"Replay exceeded {max_decisions} decisions")
            if game.pending():
                raise ValueError("Replay actions ended before the game finished")
            result = game.result()
        finally:
            game.close()
    if result is None:
        return GameSummary(
            seed_key=seed_key,
            map_name=map_name,
            ok=False,
            generations=None,
            decisions=decisions,
            error_type="MissingResultError",
            error="replay finished without a result",
        )
    return GameSummary(
        seed_key=seed_key,
        map_name=map_name,
        ok=True,
        generations=int(result.generations),
        decisions=decisions,
        scores=(int(result.scores[0]), int(result.scores[1])),
        winner=None if result.winner is None else int(result.winner),
        tie=bool(result.tie),
    )


def determinism_check(
    original: GameSummary,
    actions: Sequence[tuple[int, int]],
) -> DeterminismResult:
    """Replay and compare one finished self-play game."""
    try:
        replay = replay_game(original.seed_key, original.map_name, actions)
    except Exception as exc:  # pragma: no cover - exercised against a broken engine
        return DeterminismResult(
            seed_key=original.seed_key,
            map_name=original.map_name,
            ok=False,
            detail=f"replay failed: {type(exc).__name__}: {exc}",
        )
    return compare_replay(original, replay)
