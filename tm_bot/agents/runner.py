"""Play loop running single or batched games through the Python-Node bridge."""

from __future__ import annotations

import copy
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from tm_bot.agents.protocol import Agent, GameRunResult, SearchAgent
from tm_bot.agents.sampler import (
    ScopedWorldSampler,
    open_world_samplers,
    release_samplers,
)
from tm_bot.env.client import WorkerClient
from tm_bot.env.errors import BridgeError
from tm_bot.env.pool import WorkerPool
from tm_bot.env.types import ActorObservation, Choice, Op, Seat

DEFAULT_MAX_DECISIONS = 20_000


@dataclass(frozen=True, slots=True)
class GameSpec:
    """Specification for a single game to execute in `play_games`."""

    seed_key: str
    map_name: str
    agents: tuple[Agent, Agent]
    seat_seeds: tuple[int | None, int | None] | None = None


@dataclass
class _ActiveSlot:
    index: int
    spec: GameSpec
    gid: str
    agent0: Agent
    agent1: Agent
    versions: tuple[str | None, str | None]
    decisions: int = 0
    latencies_0: list[float] = field(default_factory=list)
    latencies_1: list[float] = field(default_factory=list)
    searched_latencies_0: list[float] = field(default_factory=list)
    searched_latencies_1: list[float] = field(default_factory=list)


def _agent_version(agent: Agent) -> str | None:
    ver = getattr(agent, "version", None)
    return str(ver) if ver is not None else None


def _validate_candidate_index(candidate_idx: Any, num_candidates: int, seat: int) -> int:
    if (
        not isinstance(candidate_idx, int)
        or isinstance(candidate_idx, bool)
        or candidate_idx < 0
        or candidate_idx >= num_candidates
    ):
        raise IndexError(
            f"Agent at seat {seat} returned out-of-range candidate index "
            f"{candidate_idx!r} for {num_candidates} candidates"
        )
    return candidate_idx


def _close_clones_quietly(sampler: ScopedWorldSampler) -> None:
    """Close a sampler's search clones without masking the caller's exception."""
    try:
        sampler.close_all()
    except Exception:
        pass


def _make_run_result(
    *,
    seed_key: str,
    map_name: str,
    result: Any,
    generations: int | None,
    decisions: int,
    versions: tuple[str | None, str | None],
    ok: bool,
    latencies_0: Sequence[float],
    latencies_1: Sequence[float],
    searched_latencies_0: Sequence[float],
    searched_latencies_1: Sequence[float],
    error_type: str | None = None,
    error: str | None = None,
) -> GameRunResult:
    return GameRunResult(
        seed_key=seed_key,
        map=map_name,
        result=result,
        generations=generations,
        decisions=decisions,
        agent_versions=versions,
        ok=ok,
        error_type=error_type,
        error=error,
        decision_latencies_ms=(tuple(latencies_0), tuple(latencies_1)),
        searched_decision_latencies_ms=(tuple(searched_latencies_0), tuple(searched_latencies_1)),
    )


def play_game(
    agents: tuple[Agent, Agent],
    seed_key: str,
    map_name: str,
    *,
    client: WorkerClient | WorkerPool | None = None,
    seat_seeds: tuple[int | None, int | None] | None = None,
    max_decisions: int = DEFAULT_MAX_DECISIONS,
) -> GameRunResult:
    """Play one complete game between two agents through the bridge.

    Standard agents (`Agent`) receive ONLY `(observation, choice)` from
    `agent.act(obs, choice)` — never the game session, game id, client, or the
    other seat's observation.

    Search agents implementing the optional `SearchAgent` protocol extension
    (`act_with_sampler`) receive `(observation, choice, sampler)`, where `sampler`
    is a seat-scoped `WorldSampler` exposing only determinized clones sampled
    from the acting seat's own observation. Every sampled clone is guaranteed to
    be closed at the end of the decision, even on error.
    """
    if len(agents) != 2:
        raise ValueError(f"Expected 2 agents, got {len(agents)}")

    agent0, agent1 = agents
    versions = (_agent_version(agent0), _agent_version(agent1))

    if seat_seeds is not None:
        agent0.reset(seat_seeds[0])
        agent1.reset(seat_seeds[1])
    else:
        agent0.reset(None)
        agent1.reset(None)

    decisions = 0
    latencies_0: list[float] = []
    latencies_1: list[float] = []
    searched_latencies_0: list[float] = []
    searched_latencies_1: list[float] = []
    ctx: Any = nullcontext(client) if client is not None else WorkerClient()

    try:
        with ctx as active_client:
            with active_client.open(seed_key=seed_key, map=map_name) as game:
                while seats := game.pending():
                    if decisions >= max_decisions:
                        return _make_run_result(
                            seed_key=seed_key,
                            map_name=map_name,
                            result=None,
                            generations=None,
                            decisions=decisions,
                            versions=versions,
                            ok=False,
                            error_type="StepCapError",
                            error=f"Exceeded max_decisions={max_decisions} for seed_key={seed_key!r}",
                            latencies_0=latencies_0,
                            latencies_1=latencies_1,
                            searched_latencies_0=searched_latencies_0,
                            searched_latencies_1=searched_latencies_1,
                        )
                    seat = seats[0]
                    obs = game.observe(seat)
                    choice = game.choice(seat)
                    agent = agent0 if seat == 0 else agent1
                    if isinstance(agent, SearchAgent):
                        t_act = time.perf_counter()
                        sampler = open_world_samplers(active_client, [(game.game_id, seat)])[0]
                        try:
                            raw_idx = agent.act_with_sampler(obs, choice, sampler)
                        finally:
                            release_samplers([sampler])
                        elapsed_ms = (time.perf_counter() - t_act) * 1000.0
                        was_searched = bool(getattr(agent, "last_decision_was_searched", False))
                    else:
                        t_act = time.perf_counter()
                        raw_idx = agent.act(obs, choice)
                        elapsed_ms = (time.perf_counter() - t_act) * 1000.0
                        was_searched = False

                    if seat == 0:
                        latencies_0.append(elapsed_ms)
                        if was_searched:
                            searched_latencies_0.append(elapsed_ms)
                    else:
                        latencies_1.append(elapsed_ms)
                        if was_searched:
                            searched_latencies_1.append(elapsed_ms)

                    candidate_idx = _validate_candidate_index(
                        raw_idx,
                        len(choice.candidates),
                        seat,
                    )
                    game.choose(seat, candidate_idx)
                    decisions += 1

                res = game.result()
                if res is None:
                    return _make_run_result(
                        seed_key=seed_key,
                        map_name=map_name,
                        result=None,
                        generations=None,
                        decisions=decisions,
                        versions=versions,
                        ok=False,
                        error_type="MissingResultError",
                        error=f"Game finished with empty pending() but result() is None for {seed_key!r}",
                        latencies_0=latencies_0,
                        latencies_1=latencies_1,
                        searched_latencies_0=searched_latencies_0,
                        searched_latencies_1=searched_latencies_1,
                    )
                return _make_run_result(
                    seed_key=seed_key,
                    map_name=map_name,
                    result=res,
                    generations=res.generations,
                    decisions=decisions,
                    versions=versions,
                    ok=True,
                    latencies_0=latencies_0,
                    latencies_1=latencies_1,
                    searched_latencies_0=searched_latencies_0,
                    searched_latencies_1=searched_latencies_1,
                )
    except (BridgeError, Exception) as exc:
        return _make_run_result(
            seed_key=seed_key,
            map_name=map_name,
            result=None,
            generations=None,
            decisions=decisions,
            versions=versions,
            ok=False,
            error_type=type(exc).__name__,
            error=str(exc),
            latencies_0=latencies_0,
            latencies_1=latencies_1,
            searched_latencies_0=searched_latencies_0,
            searched_latencies_1=searched_latencies_1,
        )


def _failure_from_exc(slot: _ActiveSlot, exc: BaseException) -> GameRunResult:
    return _make_run_result(
        seed_key=slot.spec.seed_key,
        map_name=slot.spec.map_name,
        result=None,
        generations=None,
        decisions=slot.decisions,
        versions=slot.versions,
        ok=False,
        error_type=type(exc).__name__,
        error=str(exc),
        latencies_0=slot.latencies_0,
        latencies_1=slot.latencies_1,
        searched_latencies_0=slot.searched_latencies_0,
        searched_latencies_1=slot.searched_latencies_1,
    )


def play_games(
    specs: Sequence[GameSpec],
    *,
    client: WorkerClient | WorkerPool | None = None,
    num_workers: int = 4,
    batch_size: int = 32,
    max_decisions: int = DEFAULT_MAX_DECISIONS,
    stop_when: Callable[[], bool] | None = None,
) -> list[GameRunResult]:
    """Play a batch of games concurrently across a WorkerPool with per-game failure isolation.

    Standard agents (`Agent`) receive ONLY `(observation, choice)` via `agent.act`.
    Search agents (`SearchAgent`) receive an actor-scoped `WorldSampler` via
    `act_with_sampler` or class-level `act_batch_with_samplers` so GPU neural
    evaluation and worker-pool rollouts batch across all concurrent games. All
    sampled clones are closed after every step even when an error occurs.

    `stop_when` is polled between steps; when it returns True the remaining active
    games are closed and reported with `error_type="DeadlineReached"`.
    """
    if not specs:
        return []
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    outcomes: list[GameRunResult | None] = [None] * len(specs)
    next_spec_idx = 0
    active: list[_ActiveSlot] = []

    if client is not None:
        ctx: Any = nullcontext(client)
    elif num_workers == 1:
        ctx = WorkerClient()
    else:
        ctx = WorkerPool(num_workers=num_workers)

    def spawn_slots(active_client: WorkerClient | WorkerPool) -> None:
        nonlocal next_spec_idx
        needed = batch_size - len(active)
        if needed <= 0 or next_spec_idx >= len(specs):
            return
        batch_indices = list(range(next_spec_idx, min(len(specs), next_spec_idx + needed)))
        next_spec_idx += len(batch_indices)

        create_ops = [Op.create(specs[i].seed_key, specs[i].map_name) for i in batch_indices]
        create_res = active_client.batch(create_ops, raise_on_error=False)
        for idx, res in zip(batch_indices, create_res):
            spec = specs[idx]
            if len(spec.agents) != 2:
                outcomes[idx] = GameRunResult(
                    seed_key=spec.seed_key,
                    map=spec.map_name,
                    result=None,
                    generations=None,
                    decisions=0,
                    agent_versions=(None, None),
                    ok=False,
                    error_type="ValueError",
                    error=f"Expected 2 agents, got {len(spec.agents)}",
                )
                continue
            a0 = copy.deepcopy(spec.agents[0])
            a1 = copy.deepcopy(spec.agents[1])
            versions = (_agent_version(a0), _agent_version(a1))
            if not res.ok:
                exc = res.error or RuntimeError("Unknown create error")
                outcomes[idx] = GameRunResult(
                    seed_key=spec.seed_key,
                    map=spec.map_name,
                    result=None,
                    generations=None,
                    decisions=0,
                    agent_versions=versions,
                    ok=False,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                continue
            if spec.seat_seeds is not None:
                a0.reset(spec.seat_seeds[0])
                a1.reset(spec.seat_seeds[1])
            else:
                a0.reset(None)
                a1.reset(None)
            active.append(
                _ActiveSlot(
                    index=idx,
                    spec=spec,
                    gid=str(res.value),
                    agent0=a0,
                    agent1=a1,
                    versions=versions,
                )
            )

    with ctx as active_client:
        try:
            spawn_slots(active_client)
            while active:
                if stop_when is not None and stop_when():
                    for slot in active:
                        outcomes[slot.index] = _make_run_result(
                            seed_key=slot.spec.seed_key,
                            map_name=slot.spec.map_name,
                            result=None,
                            generations=None,
                            decisions=slot.decisions,
                            versions=slot.versions,
                            ok=False,
                            error_type="DeadlineReached",
                            error=f"wall-clock deadline reached for seed_key={slot.spec.seed_key!r}",
                            latencies_0=slot.latencies_0,
                            latencies_1=slot.latencies_1,
                            searched_latencies_0=slot.searched_latencies_0,
                            searched_latencies_1=slot.searched_latencies_1,
                        )
                    try:
                        active_client.batch(
                            [Op.close(s.gid) for s in active], raise_on_error=False
                        )
                    except Exception:
                        pass
                    active = []
                    break
                pending_res = active_client.batch(
                    [Op.pending(s.gid) for s in active],
                    raise_on_error=False,
                )

                finished: list[_ActiveSlot] = []
                to_close_failed: list[_ActiveSlot] = []
                to_step: list[tuple[_ActiveSlot, Seat]] = []

                for slot, pend in zip(active, pending_res):
                    if not pend.ok:
                        outcomes[slot.index] = _failure_from_exc(
                            slot, pend.error or RuntimeError("pending() failed")
                        )
                        to_close_failed.append(slot)
                        continue
                    seats: list[Seat] = pend.value
                    if not seats:
                        finished.append(slot)
                    elif slot.decisions >= max_decisions:
                        outcomes[slot.index] = _make_run_result(
                            seed_key=slot.spec.seed_key,
                            map_name=slot.spec.map_name,
                            result=None,
                            generations=None,
                            decisions=slot.decisions,
                            versions=slot.versions,
                            ok=False,
                            error_type="StepCapError",
                            error=f"Exceeded max_decisions={max_decisions} for seed_key={slot.spec.seed_key!r}",
                            latencies_0=slot.latencies_0,
                            latencies_1=slot.latencies_1,
                            searched_latencies_0=slot.searched_latencies_0,
                            searched_latencies_1=slot.searched_latencies_1,
                        )
                        to_close_failed.append(slot)
                    else:
                        to_step.append((slot, seats[0]))

                if finished:
                    res_ops = [Op.result(s.gid) for s in finished]
                    close_ops = [Op.close(s.gid) for s in finished]
                    fin_res = active_client.batch(res_ops + close_ops, raise_on_error=False)
                    for i, slot in enumerate(finished):
                        r = fin_res[i]
                        if not r.ok:
                            outcomes[slot.index] = _failure_from_exc(
                                slot, r.error or RuntimeError("result() failed")
                            )
                        elif r.value is None:
                            outcomes[slot.index] = _make_run_result(
                                seed_key=slot.spec.seed_key,
                                map_name=slot.spec.map_name,
                                result=None,
                                generations=None,
                                decisions=slot.decisions,
                                versions=slot.versions,
                                ok=False,
                                error_type="MissingResultError",
                                error=f"result() is None for finished game {slot.spec.seed_key!r}",
                                latencies_0=slot.latencies_0,
                                latencies_1=slot.latencies_1,
                                searched_latencies_0=slot.searched_latencies_0,
                                searched_latencies_1=slot.searched_latencies_1,
                            )
                        else:
                            outcomes[slot.index] = _make_run_result(
                                seed_key=slot.spec.seed_key,
                                map_name=slot.spec.map_name,
                                result=r.value,
                                generations=r.value.generations,
                                decisions=slot.decisions,
                                versions=slot.versions,
                                ok=True,
                                latencies_0=slot.latencies_0,
                                latencies_1=slot.latencies_1,
                                searched_latencies_0=slot.searched_latencies_0,
                                searched_latencies_1=slot.searched_latencies_1,
                            )

                if to_close_failed:
                    active_client.batch(
                        [Op.close(s.gid) for s in to_close_failed],
                        raise_on_error=False,
                    )

                still_active: list[_ActiveSlot] = []
                if to_step:
                    n_step = len(to_step)
                    obs_ops = [Op.observe(s.gid, seat) for s, seat in to_step]
                    choice_ops = [Op.choice(s.gid, seat) for s, seat in to_step]
                    step_res = active_client.batch(obs_ops + choice_ops, raise_on_error=False)

                    choose_targets: list[tuple[_ActiveSlot, Seat, int]] = []
                    step_failed: list[_ActiveSlot] = []
                    batched_search_groups: dict[
                        tuple[type[Any], Any],
                        list[tuple[_ActiveSlot, Seat, Agent, ActorObservation, Choice, ScopedWorldSampler]],
                    ] = {}

                    search_entries: list[tuple[_ActiveSlot, Seat, Agent, ActorObservation, Choice]] = []
                    for i, (slot, seat) in enumerate(to_step):
                        obs_r = step_res[i]
                        ch_r = step_res[n_step + i]
                        if not obs_r.ok:
                            outcomes[slot.index] = _failure_from_exc(
                                slot, obs_r.error or RuntimeError("observe() failed")
                            )
                            step_failed.append(slot)
                            continue
                        if not ch_r.ok:
                            outcomes[slot.index] = _failure_from_exc(
                                slot, ch_r.error or RuntimeError("choice() failed")
                            )
                            step_failed.append(slot)
                            continue

                        agent = slot.agent0 if seat == 0 else slot.agent1
                        obs_val: ActorObservation = obs_r.value
                        ch_val: Choice = ch_r.value

                        if isinstance(agent, SearchAgent):
                            search_entries.append((slot, seat, agent, obs_val, ch_val))
                        else:
                            try:
                                t_act = time.perf_counter()
                                raw_idx = agent.act(obs_val, ch_val)
                                elapsed_ms = (time.perf_counter() - t_act) * 1000.0
                                if seat == 0:
                                    slot.latencies_0.append(elapsed_ms)
                                else:
                                    slot.latencies_1.append(elapsed_ms)
                                idx = _validate_candidate_index(
                                    raw_idx,
                                    len(ch_val.candidates),
                                    seat,
                                )
                                choose_targets.append((slot, seat, idx))
                            except Exception as exc:
                                outcomes[slot.index] = _failure_from_exc(slot, exc)
                                step_failed.append(slot)

                    search_samplers = open_world_samplers(
                        active_client, [(e[0].gid, e[1]) for e in search_entries]
                    )
                    for (slot, seat, agent, obs_val, ch_val), sampler in zip(
                        search_entries, search_samplers
                    ):
                        batch_fn = getattr(type(agent), "act_batch_with_samplers", None)
                        if callable(batch_fn):
                            key_fn = getattr(agent, "search_batch_key", None)
                            key = (type(agent), key_fn() if callable(key_fn) else None)
                            batched_search_groups.setdefault(key, []).append(
                                (slot, seat, agent, obs_val, ch_val, sampler)
                            )
                        else:
                            try:
                                t_act = time.perf_counter()
                                try:
                                    raw_idx = agent.act_with_sampler(obs_val, ch_val, sampler)
                                finally:
                                    release_samplers([sampler])
                                elapsed_ms = (time.perf_counter() - t_act) * 1000.0
                                was_searched = bool(
                                    getattr(agent, "last_decision_was_searched", False)
                                )
                                if seat == 0:
                                    slot.latencies_0.append(elapsed_ms)
                                    if was_searched:
                                        slot.searched_latencies_0.append(elapsed_ms)
                                else:
                                    slot.latencies_1.append(elapsed_ms)
                                    if was_searched:
                                        slot.searched_latencies_1.append(elapsed_ms)
                                idx = _validate_candidate_index(
                                    raw_idx,
                                    len(ch_val.candidates),
                                    seat,
                                )
                                choose_targets.append((slot, seat, idx))
                            except Exception as exc:
                                outcomes[slot.index] = _failure_from_exc(slot, exc)
                                step_failed.append(slot)

                    for (agent_cls, _), group in batched_search_groups.items():
                        samplers = [entry[5] for entry in group]
                        batch_exc: Exception | None = None
                        indices: list[int] | None = None
                        try:
                            indices = agent_cls.act_batch_with_samplers(
                                [entry[2] for entry in group],
                                [entry[3] for entry in group],
                                [entry[4] for entry in group],
                                samplers,
                            )
                            if len(indices) != len(group):
                                raise RuntimeError(
                                    f"act_batch_with_samplers returned {len(indices)} actions for {len(group)} requests"
                                )
                        except Exception as exc:
                            batch_exc = exc
                        finally:
                            for s in samplers:
                                _close_clones_quietly(s)

                        if batch_exc is None:
                            assert indices is not None
                            for (slot, seat, agent, _, ch_val, _), raw_idx in zip(group, indices):
                                try:
                                    elapsed_ms = float(getattr(agent, "last_decision_latency_ms", 0.0))
                                    was_searched = bool(
                                        getattr(agent, "last_decision_was_searched", False)
                                    )
                                    if seat == 0:
                                        slot.latencies_0.append(elapsed_ms)
                                        if was_searched:
                                            slot.searched_latencies_0.append(elapsed_ms)
                                    else:
                                        slot.latencies_1.append(elapsed_ms)
                                        if was_searched:
                                            slot.searched_latencies_1.append(elapsed_ms)
                                    idx = _validate_candidate_index(
                                        raw_idx,
                                        len(ch_val.candidates),
                                        seat,
                                    )
                                    choose_targets.append((slot, seat, idx))
                                except Exception as exc:
                                    outcomes[slot.index] = _failure_from_exc(slot, exc)
                                    step_failed.append(slot)
                        elif len(group) == 1:
                            slot = group[0][0]
                            outcomes[slot.index] = _failure_from_exc(slot, batch_exc)
                            step_failed.append(slot)
                        else:
                            for slot, seat, agent, obs_val, ch_val, sampler in group:
                                try:
                                    t_act = time.perf_counter()
                                    try:
                                        raw_idx = agent.act_with_sampler(obs_val, ch_val, sampler)
                                    finally:
                                        _close_clones_quietly(sampler)
                                    elapsed_ms = (time.perf_counter() - t_act) * 1000.0
                                    was_searched = bool(
                                        getattr(agent, "last_decision_was_searched", False)
                                    )
                                    if seat == 0:
                                        slot.latencies_0.append(elapsed_ms)
                                        if was_searched:
                                            slot.searched_latencies_0.append(elapsed_ms)
                                    else:
                                        slot.latencies_1.append(elapsed_ms)
                                        if was_searched:
                                            slot.searched_latencies_1.append(elapsed_ms)
                                    idx = _validate_candidate_index(
                                        raw_idx,
                                        len(ch_val.candidates),
                                        seat,
                                    )
                                    choose_targets.append((slot, seat, idx))
                                except Exception as exc:
                                    outcomes[slot.index] = _failure_from_exc(slot, exc)
                                    step_failed.append(slot)
                        release_samplers(samplers)

                    if choose_targets:
                        choose_res = active_client.batch(
                            [Op.choose(s.gid, seat, idx) for s, seat, idx in choose_targets],
                            raise_on_error=False,
                        )
                        for (slot, _, _), ch_res in zip(choose_targets, choose_res):
                            if ch_res.ok:
                                slot.decisions += 1
                                still_active.append(slot)
                            else:
                                outcomes[slot.index] = _failure_from_exc(
                                    slot, ch_res.error or RuntimeError("choose() failed")
                                )
                                step_failed.append(slot)

                    if step_failed:
                        active_client.batch(
                            [Op.close(s.gid) for s in step_failed],
                            raise_on_error=False,
                        )

                active = still_active
                spawn_slots(active_client)
        finally:
            if active:
                try:
                    active_client.batch([Op.close(s.gid) for s in active], raise_on_error=False)
                except Exception:
                    pass

    return [
        out
        if out is not None
        else GameRunResult(
            seed_key=specs[i].seed_key,
            map=specs[i].map_name,
            result=None,
            generations=None,
            decisions=0,
            agent_versions=(None, None),
            ok=False,
            error_type="InternalRunnerError",
            error="Game slot finished without an outcome",
        )
        for i, out in enumerate(outcomes)
    ]
