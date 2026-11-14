"""Determinized IS-MCTS search agent with GPU-batched neural PUCT."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Sequence

import numpy as np
import torch

from tm_bot.agents.protocol import WorldSampler
from tm_bot.agents.sampler import batch_sample_worlds, batch_world_ops
from tm_bot.env.types import ActorObservation, Choice, Op, Seat
from tm_bot.models.agent import PolicyValueAgent
from tm_bot.models.network import TMNetwork
from tm_bot.models.vocab import CardVocab
from tm_bot.search.config import SearchConfig, is_search_decision

DEFAULT_WARMSTART_CHECKPOINT = Path(r"C:\2026\tm_bot\runs\warmstart\best\checkpoint.pt")
_CHECKPOINT_CACHE: dict[tuple[str, str], PolicyValueAgent] = {}


def _normalize_priors(raw_probs: np.ndarray, num_actions: int) -> np.ndarray:
    probs = np.asarray(raw_probs[:num_actions], dtype=np.float64)
    probs = np.where(np.isfinite(probs) & (probs > 0.0), probs, 0.0)
    total = float(probs.sum())
    if total <= 0.0:
        return np.full((num_actions,), 1.0 / max(1, num_actions), dtype=np.float64)
    return probs / total


def _select_from_probs(
    probs: np.ndarray,
    num_actions: int,
    *,
    temperature: float,
    rng: random.Random,
) -> int:
    if num_actions <= 1:
        return 0
    clean = _normalize_priors(probs, num_actions)
    if temperature <= 0.0:
        return int(np.argmax(clean))
    if temperature != 1.0:
        logits = np.log(np.clip(clean, 1e-12, None)) / temperature
        logits -= logits.max()
        w = np.exp(logits)
        w = w / w.sum()
    else:
        w = clean
    return int(rng.choices(range(num_actions), weights=w.tolist(), k=1)[0])


@dataclass
class _TreeNode:
    gid: str | None
    seat: Seat
    num_actions: int
    priors: np.ndarray
    N: np.ndarray
    W: np.ndarray
    VL: np.ndarray
    total_N: float = 0.0
    is_terminal: bool = False
    terminal_value: float = 0.0
    children: dict[int, _TreeNode] = field(default_factory=dict)


def _select_puct(
    node: _TreeNode,
    *,
    sign: float,
    c_puct: float,
    fpu_value: float,
) -> int:
    n_eff = node.N + node.VL
    vl_sum = float(node.VL.sum())
    sqrt_total = math.sqrt(max(1.0, node.total_N + vl_sum))
    w_eff = node.W - node.VL if sign > 0.0 else node.W + node.VL
    q = np.where(n_eff > 0.0, sign * (w_eff / np.maximum(n_eff, 1.0)), fpu_value)
    u = c_puct * node.priors * sqrt_total / (1.0 + n_eff)
    return int(np.argmax(q + u))


@dataclass
class _SearchState:
    req_idx: int
    agent: ISMCTSAgent
    sampler: WorldSampler
    root_actor: Seat
    root_phase: str
    root: _TreeNode
    num_simulations: int
    effective_k: int
    wave_size: int
    search_seed: int
    world_root_gids: list[str] = field(default_factory=list)
    world_children: list[dict[int, _TreeNode]] = field(default_factory=list)
    open_gids: set[str] = field(default_factory=set)
    sims_done: int = 0
    elapsed_ms: float = 0.0
    sampling_failed: bool = False


@dataclass
class _SimTask:
    state: _SearchState
    world_idx: int
    path: list[tuple[_TreeNode, int]]
    parent_gid: str
    parent_seat: Seat
    parent_child_map: dict[int, _TreeNode]
    expand_action: int
    sim_gid: str = ""
    cur_seat_hint: Seat = 0
    nested_steps: int = 0
    obs: ActorObservation | None = None
    choice: Choice | None = None


class ISMCTSAgent:
    """Determinized IS-MCTS agent implementing both `Agent` and `SearchAgent`."""

    def __init__(
        self,
        policy_agent: PolicyValueAgent,
        *,
        config: SearchConfig | None = None,
        seed: int | None = None,
        version: str | None = None,
    ) -> None:
        self.policy_agent = policy_agent
        self.config = config or SearchConfig()
        self._seed = seed
        self._rng = random.Random(seed)
        self._decision_counter = 0
        self.version = version or (
            f"ismcts-s{self.config.num_simulations}-k{self.config.num_worlds}-{policy_agent.version}"
        )
        self.last_decision_was_searched: bool = False
        self.last_decision_latency_ms: float = 0.0

    @property
    def model(self) -> TMNetwork:
        return self.policy_agent.model

    @property
    def vocab(self) -> CardVocab:
        return self.policy_agent.vocab

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._seed = seed
        self._rng = random.Random(self._seed)
        self.policy_agent.reset(self._seed)
        self._decision_counter = 0
        self.last_decision_was_searched = False
        self.last_decision_latency_ms = 0.0

    def __deepcopy__(self, memo: dict[int, Any]) -> ISMCTSAgent:
        cls = self.__class__
        clone = cls.__new__(cls)
        memo[id(self)] = clone
        clone.policy_agent = copy.deepcopy(self.policy_agent, memo)
        clone.config = self.config
        clone._seed = self._seed
        clone._rng = random.Random(self._seed)
        clone._decision_counter = 0
        clone.version = self.version
        clone.last_decision_was_searched = False
        clone.last_decision_latency_ms = 0.0
        return clone

    def search_batch_key(self) -> tuple[int, int, str]:
        return (id(self.policy_agent.model), id(self.policy_agent.vocab), str(self.policy_agent.device))

    def _next_search_seed(self) -> int:
        self._decision_counter += 1
        return self._rng.getrandbits(32)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
        config: SearchConfig | None = None,
        policy_temperature: float = 0.0,
        seed: int | None = None,
        version: str | None = None,
    ) -> ISMCTSAgent:
        pv = PolicyValueAgent.from_checkpoint(
            path,
            device=device,
            temperature=policy_temperature,
            seed=seed,
        )
        return cls(pv, config=config, seed=seed, version=version)

    def act(self, observation: ActorObservation, choice: Choice) -> int:
        """Fallback `Agent.act` using the neural policy head directly when no sampler is supplied."""
        t0 = time.perf_counter()
        idx = self.policy_agent.act(observation, choice)
        self.last_decision_was_searched = False
        self.last_decision_latency_ms = (time.perf_counter() - t0) * 1000.0
        return idx

    def act_with_sampler(
        self,
        observation: ActorObservation,
        choice: Choice,
        sampler: WorldSampler,
    ) -> int:
        """Select a legal candidate index using IS-MCTS at search roots and policy head on sub-choices."""
        return self.act_batch_with_samplers(
            [self], [observation], [choice], [sampler]
        )[0]

    @classmethod
    def act_batch_with_samplers(
        cls,
        agents: Sequence[ISMCTSAgent],
        observations: Sequence[ActorObservation],
        choices: Sequence[Choice],
        samplers: Sequence[WorldSampler],
    ) -> list[int]:
        """Run batched neural inference and synchronized IS-MCTS waves across concurrent games."""
        n_reqs = len(agents)
        if n_reqs == 0:
            return []
        if not (len(observations) == len(choices) == len(samplers) == n_reqs):
            raise ValueError("Mismatched batch lengths passed to act_batch_with_samplers")

        pv_ref = agents[0].policy_agent
        outputs = [0] * n_reqs

        trivial_indices: list[int] = []
        policy_indices: list[int] = []
        search_indices: list[int] = []

        t_part0 = time.perf_counter()
        for i in range(n_reqs):
            n_cand = len(choices[i].candidates)
            if n_cand == 0:
                raise ValueError(f"Choice at request {i} has 0 candidates")
            if n_cand == 1:
                trivial_indices.append(i)
            elif is_search_decision(observations[i], choices[i]):
                search_indices.append(i)
            else:
                policy_indices.append(i)
        part_ms = (time.perf_counter() - t_part0) * 1000.0

        for idx in trivial_indices:
            outputs[idx] = 0
            agents[idx].last_decision_was_searched = False
            agents[idx].last_decision_latency_ms = part_ms / max(1, n_reqs)

        gpu_indices = policy_indices + search_indices
        if not gpu_indices:
            return outputs

        t_gpu0 = time.perf_counter()
        probs_list, _ = pv_ref.predict_batch(
            [observations[i] for i in gpu_indices],
            [choices[i] for i in gpu_indices],
            compute_value=False,
        )
        gpu_ms = (time.perf_counter() - t_gpu0) * 1000.0
        per_init_ms = gpu_ms / len(gpu_indices)

        # Direct policy head for nested sub-choices
        n_policy = len(policy_indices)
        for pos, idx in enumerate(policy_indices):
            ag = agents[idx]
            n_cand = len(choices[idx].candidates)
            phase = str(observations[idx].get("public", {}).get("phase", ""))
            eff_temp = (
                ag.config.action_temperature
                if phase == "action" and ag.config.action_temperature > 0.0
                else ag.policy_agent.temperature
            )
            outputs[idx] = _select_from_probs(
                probs_list[pos],
                n_cand,
                temperature=eff_temp,
                rng=ag._rng,
            )
            ag.last_decision_was_searched = False
            ag.last_decision_latency_ms = per_init_ms

        if not search_indices:
            return outputs

        search_states: list[_SearchState] = []
        active_states: list[_SearchState] = []
        for pos, idx in enumerate(search_indices):
            ag = agents[idx]
            cfg = ag.config
            ch = choices[idx]
            obs = observations[idx]
            n_cand = len(ch.candidates)
            priors = _normalize_priors(probs_list[n_policy + pos], n_cand)
            root = _TreeNode(
                gid=None,
                seat=int(obs["actor"]),
                num_actions=n_cand,
                priors=priors,
                N=np.zeros((n_cand,), dtype=np.float64),
                W=np.zeros((n_cand,), dtype=np.float64),
                VL=np.zeros((n_cand,), dtype=np.float64),
            )
            eff_k = min(cfg.num_worlds, cfg.num_simulations) if cfg.num_simulations > 0 else 0
            w_size = min(cfg.wave_size or max(1, eff_k), max(1, cfg.num_simulations)) if eff_k > 0 else 1
            st = _SearchState(
                req_idx=idx,
                agent=ag,
                sampler=samplers[idx],
                root_actor=int(obs["actor"]),
                root_phase=str(obs.get("public", {}).get("phase", "")),
                root=root,
                num_simulations=cfg.num_simulations,
                effective_k=eff_k,
                wave_size=w_size,
                search_seed=ag._next_search_seed(),
                elapsed_ms=per_init_ms,
            )
            search_states.append(st)
            if eff_k > 0:
                active_states.append(st)

        try:
            if active_states:
                t_samp0 = time.perf_counter()
                sampled_worlds = batch_sample_worlds(
                    [(st.sampler, st.effective_k, st.search_seed) for st in active_states]
                )
                samp_ms = (time.perf_counter() - t_samp0) * 1000.0
                per_samp_ms = samp_ms / len(active_states)
                for st, w_gids in zip(active_states, sampled_worlds):
                    st.elapsed_ms += per_samp_ms
                    if w_gids is None:
                        # No actor-consistent world could be sampled: skip the search so the
                        # unvisited root below falls back to the policy-head priors.
                        st.sampling_failed = True
                        st.sims_done = st.num_simulations
                        continue
                    st.world_root_gids = list(w_gids)
                    st.world_children = [{} for _ in w_gids]
                    st.open_gids.update(w_gids)

                while True:
                    wave_states = [st for st in active_states if st.sims_done < st.num_simulations]
                    if not wave_states:
                        break
                    t_wave0 = time.perf_counter()
                    to_expand: list[_SimTask] = []
                    ready_backups: list[tuple[_SimTask, float]] = []

                    for st in wave_states:
                        cfg = st.agent.config
                        rem = st.num_simulations - st.sims_done
                        w_batch = min(st.wave_size, rem)
                        for offset in range(w_batch):
                            w = (st.sims_done + offset) % st.effective_k
                            path: list[tuple[_TreeNode, int]] = []
                            a0 = _select_puct(
                                st.root,
                                sign=1.0,
                                c_puct=cfg.c_puct,
                                fpu_value=cfg.fpu_value,
                            )
                            st.root.VL[a0] += 1.0
                            path.append((st.root, a0))

                            child_map = st.world_children[w]
                            child = child_map.get(a0)
                            parent_gid = st.world_root_gids[w]
                            parent_seat = st.root_actor
                            expand_action = a0
                            terminal_val: float | None = None

                            while child is not None:
                                if child.is_terminal:
                                    terminal_val = child.terminal_value
                                    break
                                sign = 1.0 if child.seat == st.root_actor else -1.0
                                b = _select_puct(
                                    child,
                                    sign=sign,
                                    c_puct=cfg.c_puct,
                                    fpu_value=cfg.fpu_value,
                                )
                                child.VL[b] += 1.0
                                path.append((child, b))
                                assert child.gid is not None
                                parent_gid = child.gid
                                parent_seat = child.seat
                                child_map = child.children
                                expand_action = b
                                child = child_map.get(b)

                            task = _SimTask(
                                state=st,
                                world_idx=w,
                                path=path,
                                parent_gid=parent_gid,
                                parent_seat=parent_seat,
                                parent_child_map=child_map,
                                expand_action=expand_action,
                            )
                            if terminal_val is not None:
                                ready_backups.append((task, terminal_val))
                            else:
                                to_expand.append(task)

                        st.sims_done += w_batch

                    if to_expand:
                        cloned_res = batch_world_ops(
                            [(t.state.sampler, [Op.clone(t.parent_gid)]) for t in to_expand]
                        )
                        for t, res_pair in zip(to_expand, cloned_res):
                            t.sim_gid = str(res_pair[0])
                            t.state.open_gids.add(t.sim_gid)
                            t.cur_seat_hint = t.parent_seat

                        choose_pend_res = batch_world_ops(
                            [
                                (
                                    t.state.sampler,
                                    [
                                        Op.choose(t.sim_gid, t.parent_seat, t.expand_action),
                                        Op.pending(t.sim_gid),
                                    ],
                                )
                                for t in to_expand
                            ]
                        )

                        stepping_pairs: list[tuple[_SimTask, list[Seat]]] = [
                            (t, cp[1]) for t, cp in zip(to_expand, choose_pend_res)
                        ]
                        eval_leaves: list[_SimTask] = []

                        while stepping_pairs:
                            term_tasks: list[_SimTask] = []
                            live_tasks: list[_SimTask] = []
                            for t, seats in stepping_pairs:
                                if not seats:
                                    term_tasks.append(t)
                                else:
                                    t.cur_seat_hint = (
                                        t.cur_seat_hint
                                        if t.cur_seat_hint in seats
                                        else seats[0]
                                    )
                                    live_tasks.append(t)

                            if term_tasks:
                                term_res = batch_world_ops(
                                    [
                                        (
                                            t.state.sampler,
                                            [Op.result(t.sim_gid), Op.close(t.sim_gid)],
                                        )
                                        for t in term_tasks
                                    ]
                                )
                                for t, tr in zip(term_tasks, term_res):
                                    t.state.open_gids.discard(t.sim_gid)
                                    gres = tr[0]
                                    if gres is None or gres.tie:
                                        val = 0.0
                                    elif gres.winner == t.state.root_actor:
                                        val = 1.0
                                    else:
                                        val = -1.0
                                    if t.expand_action not in t.parent_child_map:
                                        t.parent_child_map[t.expand_action] = _TreeNode(
                                            gid=None,
                                            seat=t.cur_seat_hint,
                                            num_actions=0,
                                            priors=np.empty((0,), dtype=np.float64),
                                            N=np.empty((0,), dtype=np.float64),
                                            W=np.empty((0,), dtype=np.float64),
                                            VL=np.empty((0,), dtype=np.float64),
                                            is_terminal=True,
                                            terminal_value=val,
                                        )
                                    ready_backups.append((t, val))

                            if not live_tasks:
                                break

                            obs_ch_res = batch_world_ops(
                                [
                                    (
                                        t.state.sampler,
                                        [
                                            Op.observe(t.sim_gid, t.cur_seat_hint),
                                            Op.choice(t.sim_gid, t.cur_seat_hint),
                                        ],
                                    )
                                    for t in live_tasks
                                ]
                            )

                            nested_gpu: list[_SimTask] = []
                            choose_and_continue: list[tuple[_SimTask, int]] = []
                            for t, res_pair in zip(live_tasks, obs_ch_res):
                                t.obs = res_pair[0]
                                t.choice = res_pair[1]
                                assert t.obs is not None and t.choice is not None
                                n_cand = len(t.choice.candidates)
                                if n_cand == 1:
                                    choose_and_continue.append((t, 0))
                                elif (
                                    is_search_decision(t.obs, t.choice)
                                    or t.nested_steps >= t.state.agent.config.max_nested_steps
                                ):
                                    eval_leaves.append(t)
                                else:
                                    nested_gpu.append(t)

                            if nested_gpu:
                                nested_probs, _ = pv_ref.predict_batch(
                                    [t.obs for t in nested_gpu if t.obs is not None],
                                    [t.choice for t in nested_gpu if t.choice is not None],
                                    compute_value=False,
                                )
                                for t, nprobs in zip(nested_gpu, nested_probs):
                                    assert t.choice is not None
                                    n_cand = len(t.choice.candidates)
                                    idx_nested = int(
                                        np.argmax(_normalize_priors(nprobs, n_cand))
                                    )
                                    choose_and_continue.append((t, idx_nested))

                            if choose_and_continue:
                                next_cp_res = batch_world_ops(
                                    [
                                        (
                                            t.state.sampler,
                                            [
                                                Op.choose(t.sim_gid, t.cur_seat_hint, idx_c),
                                                Op.pending(t.sim_gid),
                                            ],
                                        )
                                        for t, idx_c in choose_and_continue
                                    ]
                                )
                                stepping_pairs = []
                                for (t, _), cp in zip(choose_and_continue, next_cp_res):
                                    t.nested_steps += 1
                                    stepping_pairs.append((t, cp[1]))
                            else:
                                stepping_pairs = []

                        if eval_leaves:
                            leaf_probs, leaf_vals = pv_ref.predict_batch(
                                [t.obs for t in eval_leaves if t.obs is not None],
                                [t.choice for t in eval_leaves if t.choice is not None],
                                compute_value=True,
                            )
                            for t, lprobs, lval in zip(eval_leaves, leaf_probs, leaf_vals):
                                assert t.choice is not None
                                v_actor = (
                                    float(lval)
                                    if t.cur_seat_hint == t.state.root_actor
                                    else -float(lval)
                                )
                                n_act = len(t.choice.candidates)
                                priors = _normalize_priors(lprobs, n_act)
                                if t.expand_action not in t.parent_child_map:
                                    t.parent_child_map[t.expand_action] = _TreeNode(
                                        gid=t.sim_gid,
                                        seat=t.cur_seat_hint,
                                        num_actions=n_act,
                                        priors=priors,
                                        N=np.zeros((n_act,), dtype=np.float64),
                                        W=np.zeros((n_act,), dtype=np.float64),
                                        VL=np.zeros((n_act,), dtype=np.float64),
                                    )
                                ready_backups.append((t, v_actor))

                    for t, v_actor in ready_backups:
                        for node, act in t.path:
                            node.VL[act] = max(0.0, node.VL[act] - 1.0)
                            node.N[act] += 1.0
                            node.W[act] += v_actor
                            node.total_N += 1.0

                    wave_ms = (time.perf_counter() - t_wave0) * 1000.0
                    per_wave_ms = wave_ms / len(wave_states)
                    for st in wave_states:
                        st.elapsed_ms += per_wave_ms

            for st in search_states:
                root = st.root
                temp = (
                    st.agent.config.action_temperature
                    if st.root_phase == "action" and st.agent.config.action_temperature > 0.0
                    else st.agent.config.temperature
                )
                if root.total_N <= 0.0 or float(root.N.max()) <= 0.0:
                    chosen = _select_from_probs(
                        root.priors,
                        root.num_actions,
                        temperature=temp,
                        rng=st.agent._rng,
                    )
                elif temp <= 0.0:
                    best_n = root.N.max()
                    tied = np.flatnonzero(root.N == best_n)
                    if len(tied) == 1:
                        chosen = int(tied[0])
                    else:
                        chosen = int(
                            max(
                                tied.tolist(),
                                key=lambda a: (
                                    float(root.W[a] / max(1.0, root.N[a])),
                                    float(root.priors[a]),
                                    -a,
                                ),
                            )
                        )
                else:
                    scaled = np.power(np.maximum(root.N, 0.0), 1.0 / temp)
                    total_s = float(scaled.sum())
                    if total_s <= 0.0 or not np.isfinite(total_s):
                        chosen = int(np.argmax(root.N))
                    else:
                        probs_s = (scaled / total_s).tolist()
                        chosen = int(
                            st.agent._rng.choices(
                                range(root.num_actions), weights=probs_s, k=1
                            )[0]
                        )

                outputs[st.req_idx] = chosen
                st.agent.last_decision_was_searched = not st.sampling_failed
                st.agent.last_decision_latency_ms = st.elapsed_ms

        finally:
            close_pairs = [
                (st.sampler, [Op.close(gid) for gid in st.open_gids])
                for st in active_states
                if st.open_gids
            ]
            if close_pairs:
                try:
                    batch_world_ops(close_pairs, raise_on_error=False)
                except Exception:
                    pass
            for st in active_states:
                st.open_gids.clear()
                try:
                    st.sampler.close_all()
                except Exception:
                    pass

        return outputs


def make_warmstart_search_agent(
    *,
    checkpoint: str | Path | None = None,
    device: str | None = None,
    num_simulations: int | None = None,
    num_worlds: int | None = None,
    c_puct: float | None = None,
    temperature: float | None = None,
    action_temperature: float | None = None,
    wave_size: int | None = None,
    seed: int | None = None,
) -> ISMCTSAgent:
    """Factory loading and caching the warm-start checkpoint for eval/search runs."""
    ckpt_path = str(
        checkpoint
        or os.environ.get("TM_SEARCH_CHECKPOINT")
        or DEFAULT_WARMSTART_CHECKPOINT
    )
    resolved_device = (
        device
        or os.environ.get("TM_SEARCH_DEVICE")
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    cache_key = (str(Path(ckpt_path).resolve()), resolved_device)
    pv = _CHECKPOINT_CACHE.get(cache_key)
    if pv is None:
        pv = PolicyValueAgent.from_checkpoint(ckpt_path, device=resolved_device, temperature=0.0)
        _CHECKPOINT_CACHE[cache_key] = pv

    sims = (
        num_simulations
        if num_simulations is not None
        else int(os.environ.get("TM_SEARCH_SIMS", "8"))
    )
    worlds = (
        num_worlds
        if num_worlds is not None
        else int(os.environ.get("TM_SEARCH_WORLDS", "4"))
    )
    cpuct = (
        c_puct
        if c_puct is not None
        else float(os.environ.get("TM_SEARCH_C_PUCT", "1.5"))
    )
    temp = (
        temperature
        if temperature is not None
        else float(os.environ.get("TM_SEARCH_TEMP", "0.0"))
    )
    act_temp = (
        action_temperature
        if action_temperature is not None
        else float(os.environ.get("TM_SEARCH_ACTION_TEMP", "1.0"))
    )
    wsize_env = os.environ.get("TM_SEARCH_WAVE_SIZE")
    wsize = (
        wave_size
        if wave_size is not None
        else (int(wsize_env) if wsize_env else 4)
    )
    cfg = SearchConfig(
        num_simulations=sims,
        num_worlds=worlds,
        c_puct=cpuct,
        temperature=temp,
        action_temperature=act_temp,
        wave_size=wsize,
    )
    return ISMCTSAgent(pv, config=cfg, seed=seed)
