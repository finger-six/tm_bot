"""Scoped simulation handle (`WorldSampler`) for search agents.

The runner never hands a search agent the live game. For each searched decision
it first creates a *belief root*: a `determinize_clone` of the live decision
point whose hidden state (opponent hand, deck order, ...) is sampled from the
acting seat's own observation. The sampler only knows that root's id and the
clones it creates from it, so everything reachable through it is belief-derived;
the true hidden state and the live game id are not stored on the sampler.

This is a structural guard against accidental leakage by in-process agent code,
not a security boundary: Python code can always introspect the worker pool.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from tm_bot.agents.protocol import WorldSampler
from tm_bot.env.client import WorkerClient
from tm_bot.env.pool import WorkerPool
from tm_bot.env.types import ActorObservation, Choice, GameResult, Op, Seat

_LOG = logging.getLogger(__name__)

_ALLOWED_CLONE_METHODS = frozenset(
    {"pending", "observe", "choice", "choose", "result", "clone", "close"}
)

# Seed for the runner-created belief root. Its sampled hidden state never influences
# search: every world is re-sampled from the actor observation, which the root shares
# with the live game.
_BELIEF_ROOT_SEED = 0


class SamplerUnavailableError(RuntimeError):
    """Raised when no actor-consistent world could be sampled for this decision."""


def _derive_world_seeds(seed: int, k: int) -> list[int]:
    """Derive `k` deterministic 32-bit unsigned world seeds via SplitMix32."""
    seeds: list[int] = []
    state = (int(seed) + 0x9E3779B9) & 0xFFFFFFFF
    for _ in range(k):
        state = (state + 0x9E3779B9) & 0xFFFFFFFF
        z = state
        z = ((z ^ (z >> 16)) * 0x85EBCA6B) & 0xFFFFFFFF
        z = ((z ^ (z >> 13)) * 0xC2B2AE35) & 0xFFFFFFFF
        z = (z ^ (z >> 16)) & 0xFFFFFFFF
        seeds.append(z)
    return seeds


class ScopedWorldSampler:
    """Actor-scoped `WorldSampler` bound to a belief root, never to the live game.

    Build instances with `open_world_samplers`; release them with `release_samplers`.
    Every clone-targeting method validates that the target id is a clone owned by
    this sampler.
    """

    __slots__ = ("_seat", "_client", "_root_id", "_root_error", "_owned_ids")

    def __init__(
        self,
        client: WorkerClient | WorkerPool,
        belief_root_id: str | None,
        seat: Seat,
        *,
        root_error: str | None = None,
    ) -> None:
        if seat not in (0, 1):
            raise ValueError(f"seat must be 0 or 1, got {seat!r}")
        if belief_root_id is None and root_error is None:
            raise ValueError("either belief_root_id or root_error is required")
        self._seat: Seat = seat
        self._client = client
        self._root_id = belief_root_id
        self._root_error = root_error
        self._owned_ids: set[str] = set()

    @property
    def seat(self) -> Seat:
        return self._seat

    def _require_root(self) -> str:
        if self._root_id is None:
            raise SamplerUnavailableError(
                f"no actor-consistent world available: {self._root_error}"
            )
        return self._root_id

    def _sample_ops(self, k: int, seed: int) -> list[Op]:
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        root = self._require_root()
        return [
            Op.determinize_clone(root, self._seat, w_seed)
            for w_seed in _derive_world_seeds(seed, k)
        ]

    def _require_owned(self, clone_id: str) -> None:
        if not isinstance(clone_id, str) or clone_id not in self._owned_ids:
            raise PermissionError(
                f"Access denied: {clone_id!r} is not a clone owned by this WorldSampler"
            )

    def _validate_op(self, op: Op) -> None:
        if op.method not in _ALLOWED_CLONE_METHODS:
            raise PermissionError(
                f"Operation {op.method!r} is not permitted via WorldSampler.batch_ops"
            )
        gid = op.params.get("gameId")
        if not isinstance(gid, str) or gid not in self._owned_ids:
            raise PermissionError(
                f"Access denied: op {op.method!r} targets unowned gameId={gid!r}"
            )

    def _track(self, ops: Sequence[Op], results: Sequence[Any], *, raised: bool) -> None:
        for op, res in zip(ops, results):
            if raised:
                ok, value = True, res
            else:
                ok, value = bool(res.ok), res.value
            if not ok:
                continue
            if op.method in ("clone", "determinize_clone"):
                self._owned_ids.add(str(value))
            elif op.method == "close":
                self._owned_ids.discard(str(op.params["gameId"]))

    def sample(self, k: int, seed: int) -> list[str]:
        ops = self._sample_ops(k, seed)
        if not ops:
            return []
        results = self._client.batch(ops, raise_on_error=False)
        self._track(ops, results, raised=False)
        failed = [r for r in results if not r.ok]
        if failed:
            self.close([str(r.value) for r in results if r.ok])
            raise SamplerUnavailableError(f"world sampling failed: {failed[0].error}")
        return [str(r.value) for r in results]

    def clone(self, clone_id: str) -> str:
        self._require_owned(clone_id)
        res = self._client.batch([Op.clone(clone_id)], raise_on_error=True)[0]
        new_cid = str(res)
        self._owned_ids.add(new_cid)
        return new_cid

    def close(self, clone_ids: Sequence[str] | str) -> None:
        targets = [clone_ids] if isinstance(clone_ids, str) else list(clone_ids)
        if not targets:
            return
        for cid in targets:
            self._require_owned(cid)
        try:
            self._client.batch([Op.close(cid) for cid in targets], raise_on_error=False)
        finally:
            for cid in targets:
                self._owned_ids.discard(cid)

    def close_all(self) -> None:
        if self._owned_ids:
            self.close(list(self._owned_ids))

    def pending(self, clone_id: str) -> list[Seat]:
        self._require_owned(clone_id)
        return self._client.pending(clone_id)

    def observe(self, clone_id: str, seat: Seat) -> ActorObservation:
        self._require_owned(clone_id)
        return self._client.observe(clone_id, seat)

    def choice(self, clone_id: str, seat: Seat) -> Choice:
        self._require_owned(clone_id)
        return self._client.choice(clone_id, seat)

    def choose(self, clone_id: str, seat: Seat, candidate_idx: int) -> None:
        self._require_owned(clone_id)
        self._client.choose(clone_id, seat, candidate_idx)

    def result(self, clone_id: str) -> GameResult | None:
        self._require_owned(clone_id)
        return self._client.result(clone_id)

    def batch_ops(
        self, ops: Sequence[Op], *, raise_on_error: bool = True
    ) -> list[Any]:
        if not ops:
            return []
        for op in ops:
            self._validate_op(op)
        results = self._client.batch(ops, raise_on_error=raise_on_error)
        self._track(ops, results, raised=raise_on_error)
        return results


def open_world_samplers(
    client: WorkerClient | WorkerPool,
    requests: Sequence[tuple[str, Seat]],
) -> list[ScopedWorldSampler]:
    """Create one belief root per `(live_game_id, seat)` in a single batch (runner-only).

    A request whose root cannot be sampled yields a sampler that raises
    `SamplerUnavailableError` from `sample`, so search agents can fall back.
    """
    if not requests:
        return []
    ops = [
        Op.determinize_clone(live_gid, seat, _BELIEF_ROOT_SEED)
        for live_gid, seat in requests
    ]
    results = client.batch(ops, raise_on_error=False)
    samplers: list[ScopedWorldSampler] = []
    for (_, seat), res in zip(requests, results):
        if res.ok:
            samplers.append(ScopedWorldSampler(client, str(res.value), seat))
        else:
            _LOG.warning("belief root sampling failed (seat %s): %s", seat, res.error)
            samplers.append(
                ScopedWorldSampler(client, None, seat, root_error=str(res.error))
            )
    return samplers


def release_samplers(samplers: Sequence[ScopedWorldSampler]) -> None:
    """Close every clone and belief root of `samplers` (runner-only); never raises."""
    by_client: dict[int, tuple[WorkerClient | WorkerPool, list[Op]]] = {}
    for s in samplers:
        gids = list(s._owned_ids)
        if s._root_id is not None:
            gids.append(s._root_id)
        s._owned_ids.clear()
        s._root_id = None
        s._root_error = "sampler released"
        if gids:
            by_client.setdefault(id(s._client), (s._client, []))[1].extend(
                Op.close(g) for g in gids
            )
    for client, ops in by_client.values():
        try:
            client.batch(ops, raise_on_error=False)
        except Exception:
            pass


def batch_sample_worlds(
    requests: Sequence[tuple[WorldSampler, int, int]],
) -> list[list[str] | None]:
    """Sample worlds for many samplers, batching per worker client when possible.

    Returns one list of clone ids per request, or `None` when that request's
    worlds could not be sampled (its partial clones are already closed).
    """
    if not requests:
        return []
    out: list[list[str] | None] = [None] * len(requests)

    groups: dict[int, list[int]] = {}
    for i, (sampler, _, _) in enumerate(requests):
        if isinstance(sampler, ScopedWorldSampler):
            groups.setdefault(id(sampler._client), []).append(i)
        else:
            try:
                out[i] = sampler.sample(requests[i][1], requests[i][2])
            except Exception:
                out[i] = None

    for idxs in groups.values():
        flat_ops: list[Op] = []
        spans: list[tuple[int, int, int]] = []
        for i in idxs:
            sampler, k, seed = requests[i]
            assert isinstance(sampler, ScopedWorldSampler)
            try:
                ops = sampler._sample_ops(k, seed)
            except SamplerUnavailableError:
                continue
            spans.append((i, len(flat_ops), len(ops)))
            flat_ops.extend(ops)
        if not flat_ops:
            for i, _, _ in spans:
                out[i] = []
            continue
        client = requests[idxs[0]][0]._client  # type: ignore[union-attr]
        results = client.batch(flat_ops, raise_on_error=False)
        for i, start, count in spans:
            sampler = requests[i][0]
            assert isinstance(sampler, ScopedWorldSampler)
            chunk_ops = flat_ops[start : start + count]
            chunk = results[start : start + count]
            sampler._track(chunk_ops, chunk, raised=False)
            if all(r.ok for r in chunk):
                out[i] = [str(r.value) for r in chunk]
            else:
                err = next(r.error for r in chunk if not r.ok)
                _LOG.warning("world sampling failed (seat %s): %s", sampler.seat, err)
                sampler.close([str(r.value) for r in chunk if r.ok])
                out[i] = None
    return out


def batch_world_ops(
    pairs: Sequence[tuple[WorldSampler, Sequence[Op]]],
    *,
    raise_on_error: bool = True,
) -> list[list[Any]]:
    """Execute clone operations across multiple samplers in one worker-pool batch when possible."""
    if not pairs:
        return []

    if all(isinstance(s, ScopedWorldSampler) for s, _ in pairs):
        first_client = pairs[0][0]._client  # type: ignore[union-attr]
        if all(s._client is first_client for s, _ in pairs):  # type: ignore[union-attr]
            flat_ops: list[Op] = []
            for sampler, ops in pairs:
                assert isinstance(sampler, ScopedWorldSampler)
                for op in ops:
                    sampler._validate_op(op)
                flat_ops.extend(ops)
            if not flat_ops:
                return [[] for _ in pairs]
            flat_results = first_client.batch(flat_ops, raise_on_error=raise_on_error)
            grouped: list[list[Any]] = []
            offset = 0
            for sampler, ops in pairs:
                assert isinstance(sampler, ScopedWorldSampler)
                sub_res = list(flat_results[offset : offset + len(ops)])
                sampler._track(ops, sub_res, raised=raise_on_error)
                grouped.append(sub_res)
                offset += len(ops)
            return grouped

    return [sampler.batch_ops(ops, raise_on_error=raise_on_error) for sampler, ops in pairs]
