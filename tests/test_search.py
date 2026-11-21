from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest
import torch

from tm_bot.agents import B0Agent, B1Agent, GameSpec, play_game, play_games
from tm_bot.agents.card_data import load_card_data
from tm_bot.agents.protocol import Agent, GameRunResult, SearchAgent, WorldSampler
from tm_bot.agents.sampler import (
    SamplerUnavailableError,
    ScopedWorldSampler,
    open_world_samplers,
    release_samplers,
)
from tm_bot.env import Candidate, Choice, Op, WorkerClient, WorkerPool
from tm_bot.eval import evaluate_agents, format_markdown_report
from tm_bot.models import CARD_STATIC_DIM, CardVocab, PolicyValueAgent, TMModelConfig, TMNetwork
from tm_bot.search import ISMCTSAgent, SearchConfig, is_search_decision


def _make_tiny_policy_value_agent(*, seed: int = 0) -> PolicyValueAgent:
    cards = load_card_data()
    vocab = CardVocab.from_card_data(cards)
    cfg = TMModelConfig(
        vocab_size=vocab.size,
        d_model=32,
        card_id_emb_dim=16,
        card_static_proj_dim=16,
        slot_emb_dim=8,
        card_dyn_proj_dim=8,
        space_hidden_dim=16,
        n_heads=2,
        n_obs_layers=1,
        dim_feedforward=64,
        policy_ff_dim=32,
        dropout=0.0,
    )
    static_matrix = torch.from_numpy(vocab.static_features.copy())
    torch.manual_seed(seed)
    model = TMNetwork(cfg, static_matrix)
    assert static_matrix.shape[1] == CARD_STATIC_DIM
    return PolicyValueAgent(
        model=model,
        vocab=vocab,
        device="cpu",
        temperature=0.0,
        seed=seed,
        version="tiny-test-v1",
    )


def _make_dummy_obs(
    *,
    actor: int = 0,
    phase: str = "action",
) -> dict[str, Any]:
    def _player(seat: int) -> dict[str, Any]:
        return {
            "seat": seat,
            "color": "red" if seat == 0 else "blue",
            "terraformRating": 20,
            "megacredits": 25,
            "steel": 1,
            "titanium": 0,
            "plants": 0,
            "energy": 0,
            "heat": 0,
            "megacreditProduction": 0,
            "steelProduction": 0,
            "titaniumProduction": 0,
            "plantProduction": 0,
            "energyProduction": 0,
            "heatProduction": 0,
            "steelValue": 2,
            "titaniumValue": 3,
            "cardCost": 3,
            "cardDiscount": 0,
            "cardsInHandCount": 2,
            "citiesCount": 0,
            "noTagsCount": 0,
            "tags": {},
            "tableau": [],
            "usedActions": [],
            "actionsTakenThisRound": 0,
            "actionsTakenThisGame": 0,
            "availableBlueCardActionCount": 0,
            "lastCardPlayed": None,
            "isActive": seat == actor,
            "passed": False,
            "protectedResources": {},
            "protectedProduction": {},
        }

    return {
        "actor": actor,
        "public": {
            "map": "tharsis",
            "generation": 2,
            "phase": phase,
            "firstPlayer": 0,
            "activePlayer": actor,
            "passedPlayers": [],
            "globalParameters": {
                "temperature": -24,
                "oxygenLevel": 2,
                "oceans": 1,
                "isTerraformed": False,
            },
            "deckSize": 120,
            "discardPileSize": 0,
            "otherDeckSizes": {"corporations": {"drawPile": 8, "discardPile": 0}, "preludes": None},
            "spaces": [],
            "milestones": [],
            "awards": [],
            "players": [_player(0), _player(1)],
            "publicLog": [],
        },
        "private": {
            "cardsInHand": [],
            "dealtCorporationCards": [],
            "dealtPreludeCards": [],
            "dealtProjectCards": [],
            "pickedCorporationCard": "CrediCor",
            "preludeCardsInHand": [],
            "draftedCards": [],
            "currentDraftPacket": [],
        },
        "history": {
            "initialBuy": None,
            "drafts": [],
            "researchBuys": [],
            "cardSelections": [],
        },
    }


def test_is_search_decision_classification() -> None:
    c2 = (Candidate(id="a", label="A"), Candidate(id="b", label="B"))
    c1 = (Candidate(id="a", label="A"),)

    # Single candidate never searched
    assert is_search_decision(_make_dummy_obs(phase="action"), Choice(0, "branch", "Take your first action", c1)) is False

    # Action-phase root branches ARE searched; nested branches are NOT
    assert is_search_decision(_make_dummy_obs(phase="action"), Choice(0, "branch", "Take your first action", c2)) is True
    assert is_search_decision(_make_dummy_obs(phase="action"), Choice(0, "branch", "Take your next action", c2)) is True
    assert is_search_decision(_make_dummy_obs(phase="action"), Choice(0, "branch", "Select option", c2)) is False

    # Action-phase nested sub-choices use policy head directly
    for kind in ("card", "payment", "space", "player", "amount", "resource", "multi_card"):
        assert is_search_decision(_make_dummy_obs(phase="action"), Choice(0, kind, "Prompt", c2)) is False

    # Draft card picks ARE searched
    assert is_search_decision(_make_dummy_obs(phase="drafting"), Choice(0, "card", "Select a card to keep", c2)) is True
    assert is_search_decision(_make_dummy_obs(phase="initial_drafting"), Choice(0, "card", "Select a card to keep", c2)) is True

    # Research/initial buy multi_card prompts ARE searched; initial corp/prelude use policy head directly
    assert is_search_decision(_make_dummy_obs(phase="research"), Choice(0, "multi_card", "Select initial cards to buy", c2)) is True
    assert is_search_decision(_make_dummy_obs(phase="research"), Choice(0, "multi_card", "Select cards to buy", c2)) is True
    assert is_search_decision(_make_dummy_obs(phase="research"), Choice(0, "card", "Select corporation", c2)) is False
    assert is_search_decision(_make_dummy_obs(phase="research"), Choice(0, "multi_card", "Select 2 Prelude cards", c2)) is False


def test_world_sampler_isolation_and_observation_only_determinization() -> None:
    with WorkerClient() as client:
        with client.open("search-sampler-iso-1", "tharsis") as live:
            obs0_live = live.observe(0)
            obs1_live = live.observe(1)
            sampler = open_world_samplers(client, [(live.game_id, 0)])[0]
            assert isinstance(sampler, WorldSampler)
            assert sampler.seat == 0

            # Rejects direct operations on the live game or foreign clone ids
            for bad_gid in (live.game_id, "foreign-game-id"):
                with pytest.raises(PermissionError):
                    sampler.clone(bad_gid)
                with pytest.raises(PermissionError):
                    sampler.close(bad_gid)
                with pytest.raises(PermissionError):
                    sampler.pending(bad_gid)
                with pytest.raises(PermissionError):
                    sampler.observe(bad_gid, 1)
                with pytest.raises(PermissionError):
                    sampler.choice(bad_gid, 0)
                with pytest.raises(PermissionError):
                    sampler.choose(bad_gid, 0, 0)
                with pytest.raises(PermissionError):
                    sampler.result(bad_gid)
                with pytest.raises(PermissionError):
                    sampler.batch_ops([Op.observe(bad_gid, 1)])

            # sample(k, seed) determinizes from actor obs only and is deterministic
            clones_a = sampler.sample(3, seed=12345)
            clones_b = sampler.sample(3, seed=12345)
            assert len(clones_a) == 3
            assert len(set(clones_a)) == 3
            assert all(cid != live.game_id for cid in clones_a)

            for cid_a, cid_b in zip(clones_a, clones_b):
                assert sampler.observe(cid_a, 0) == obs0_live
                assert sampler.observe(cid_a, 1) == sampler.observe(cid_b, 1)

            # Sampled worlds differ across seeds
            assert sampler.observe(clones_a[0], 1) != sampler.observe(clones_a[1], 1)

            # Two worlds with DIFFERENT opponent hands/dealt cards but identical actor obs
            # produce byte-identical determinizations for the same seed because the sampler
            # receives only the actor observation.
            alt_world_0 = clones_a[0]
            alt_world_1 = clones_a[1]
            assert sampler.observe(alt_world_0, 0) == obs0_live
            assert sampler.observe(alt_world_1, 0) == obs0_live
            assert sampler.observe(alt_world_0, 1) != sampler.observe(alt_world_1, 1)

            sampler_w0 = ScopedWorldSampler(client, alt_world_0, 0)
            sampler_w1 = ScopedWorldSampler(client, alt_world_1, 0)
            try:
                res_w0 = sampler_w0.sample(1, seed=55555)
                res_w1 = sampler_w1.sample(1, seed=55555)
                assert sampler_w0.observe(res_w0[0], 1) == sampler_w1.observe(res_w1[0], 1)
            finally:
                sampler_w0.close_all()
                sampler_w1.close_all()

            # close_all drops search clones; release also drops the belief root
            sampler.close_all()
            assert client.ping()["games"] == 2
            release_samplers([sampler])
            assert client.ping()["games"] == 1
            with pytest.raises(SamplerUnavailableError):
                sampler.sample(1, seed=1)
            assert live.observe(0) == obs0_live
            assert live.observe(1) == obs1_live


def _reachable_state(root: object, *, opaque: tuple[type, ...]) -> list[object]:
    """Objects reachable from `root` via attributes, slots, containers and closures.

    Instances of `opaque` types (the transport) are recorded but not descended into.
    Classes, modules and code are skipped: they hold no per-decision state.
    """
    import types

    seen: set[int] = set()
    out: list[object] = []
    stack: list[object] = [root]
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        out.append(obj)
        if isinstance(obj, opaque) or isinstance(obj, (str, bytes, int, float, bool, type(None))):
            continue
        if isinstance(obj, (type, types.ModuleType, types.BuiltinFunctionType)):
            continue
        if isinstance(obj, dict):
            stack.extend(obj.keys())
            stack.extend(obj.values())
            continue
        if isinstance(obj, (list, tuple, set, frozenset)):
            stack.extend(obj)
            continue
        if isinstance(obj, types.FunctionType):
            stack.extend(c.cell_contents for c in (obj.__closure__ or ()))
            continue
        if isinstance(obj, types.MethodType):
            stack.append(obj.__func__)
            stack.append(obj.__self__)
            continue
        names: set[str] = set(getattr(obj, "__dict__", {}).keys())
        for klass in type(obj).__mro__:
            slots = klass.__dict__.get("__slots__", ())
            names.update([slots] if isinstance(slots, str) else slots)
        for name in names:
            if name in ("__weakref__", "__dict__"):
                continue
            try:
                stack.append(getattr(obj, name))
            except AttributeError:
                pass
    return out


def test_world_sampler_has_no_reachable_path_to_live_game() -> None:
    import tm_bot.agents.sampler as sampler_mod

    with WorkerClient() as client:
        with client.open("search-sampler-noleak-1", "tharsis") as live:
            seat = live.pending()[0]
            opp = 1 - seat
            sampler = open_world_samplers(client, [(live.game_id, seat)])[0]
            try:
                clones = sampler.sample(2, seed=7)
                reachable = _reachable_state(sampler, opaque=(WorkerClient, WorkerPool))
                module_state = _reachable_state(
                    {k: v for k, v in vars(sampler_mod).items() if not k.startswith("__")},
                    opaque=(WorkerClient, WorkerPool),
                )
                # The live game id is stored nowhere on the sampler or in module state.
                assert live.game_id not in [o for o in reachable if isinstance(o, str)]
                assert live.game_id not in [o for o in module_state if isinstance(o, str)]
                assert not any(isinstance(o, (WorkerClient, WorkerPool)) for o in module_state)
                assert not hasattr(sampler, "_ctx")

                # The only non-clone game the sampler knows is a belief root: same actor view,
                # opponent's hidden state re-sampled (differs from the live truth).
                root_ids = [
                    o for o in reachable
                    if isinstance(o, str) and o.startswith("g-") and o not in clones
                ]
                assert len(root_ids) == 1
                assert client.observe(root_ids[0], seat) == live.observe(seat)
                assert client.observe(root_ids[0], opp) != live.observe(opp)
            finally:
                release_samplers([sampler])
            assert client.ping()["games"] == 1

            # A root that cannot be built yields a sampler that refuses to sample (no crash).
            broken = open_world_samplers(client, [("no-such-game", seat)])[0]
            with pytest.raises(SamplerUnavailableError):
                broken.sample(2, seed=1)
            release_samplers([broken])


def test_search_falls_back_to_policy_head_when_worlds_cannot_be_sampled() -> None:
    pv = _make_tiny_policy_value_agent(seed=5)
    agent = ISMCTSAgent(pv, config=SearchConfig(num_simulations=4, num_worlds=2), seed=3)
    agent.reset(0)
    b0 = B0Agent(seed=0)
    with WorkerClient() as client:
        with client.open("search-fallback-1", "tharsis") as live:
            while True:
                seat = live.pending()[0]
                obs = live.observe(seat)
                ch = live.choice(seat)
                if len(ch.candidates) > 1 and is_search_decision(obs, ch):
                    break
                live.choose(seat, b0.act(obs, ch))

            broken = ScopedWorldSampler(client, None, seat, root_error="simulated impossible world")
            idx = agent.act_with_sampler(obs, ch, broken)
            assert 0 <= idx < len(ch.candidates)
            assert agent.last_decision_was_searched is False
            assert idx == agent.act(obs, ch)
            assert client.ping()["games"] == 1


def test_search_returns_legal_candidate_never_mutates_live_game_and_cleans_clones() -> None:
    pv = _make_tiny_policy_value_agent(seed=7)
    agent = ISMCTSAgent(
        pv,
        config=SearchConfig(num_simulations=4, num_worlds=2, c_puct=1.25),
        seed=99,
    )
    assert isinstance(agent, Agent)
    assert isinstance(agent, SearchAgent)
    agent.reset(42)

    # Verify fallback Agent.act works cleanly even without a sampler
    b0 = B0Agent(seed=1)

    searched_calls = 0
    nested_calls = 0

    with WorkerClient() as client:
        with client.open("search-live-safety-1", "tharsis") as live:
            while seats := live.pending():
                seat = seats[0]
                obs = live.observe(seat)
                choice = live.choice(seat)

                if seat == 0:
                    obs0_before = live.observe(0)
                    obs1_before = live.observe(1)
                    ch0_before = live.choice(0)
                    pend_before = live.pending()

                    sampler = open_world_samplers(client, [(live.game_id, 0)])[0]
                    sampled_before = 0
                    orig_sample = sampler.sample

                    class _CountingSampler:
                        def __init__(self, inner: ScopedWorldSampler) -> None:
                            self._inner = inner
                            self.sample_calls = 0

                        @property
                        def seat(self) -> int:
                            return self._inner.seat

                        def sample(self, k: int, seed: int) -> list[str]:
                            self.sample_calls += 1
                            return orig_sample(k, seed)

                        def clone(self, clone_id: str) -> str:
                            return self._inner.clone(clone_id)

                        def close(self, clone_ids: Sequence[str] | str) -> None:
                            return self._inner.close(clone_ids)

                        def close_all(self) -> None:
                            return self._inner.close_all()

                        def pending(self, clone_id: str) -> list[int]:
                            return self._inner.pending(clone_id)

                        def observe(self, clone_id: str, s: int) -> Any:
                            return self._inner.observe(clone_id, s)

                        def choice(self, clone_id: str, s: int) -> Choice:
                            return self._inner.choice(clone_id, s)

                        def choose(self, clone_id: str, s: int, candidate_idx: int) -> None:
                            return self._inner.choose(clone_id, s, candidate_idx)

                        def result(self, clone_id: str) -> Any:
                            return self._inner.result(clone_id)

                        def batch_ops(self, ops: Sequence[Op], *, raise_on_error: bool = True) -> Any:
                            return self._inner.batch_ops(ops, raise_on_error=raise_on_error)

                    spy_sampler = _CountingSampler(sampler)
                    try:
                        idx = agent.act_with_sampler(obs, choice, spy_sampler)
                    finally:
                        release_samplers([sampler])

                    assert isinstance(idx, int) and not isinstance(idx, bool)
                    assert 0 <= idx < len(choice.candidates)

                    # Live game is strictly untouched and zero clones leaked
                    assert live.observe(0) == obs0_before
                    assert live.observe(1) == obs1_before
                    assert live.choice(0) == ch0_before
                    assert live.pending() == pend_before
                    assert client.ping()["games"] == 1

                    if is_search_decision(obs, choice):
                        searched_calls += 1
                        assert spy_sampler.sample_calls == 1
                    else:
                        nested_calls += 1
                        assert spy_sampler.sample_calls == 0

                    _ = sampled_before
                else:
                    idx = b0.act(obs, choice)

                live.choose(seat, idx)

            res = live.result()
            assert res is not None
            assert searched_calls > 0
            assert nested_calls > 0


def test_search_leaves_back_up_the_margin_utility(monkeypatch: pytest.MonkeyPatch) -> None:
    """Leaf evaluation asks for the margin head and feeds U = (2p - 1) + 0.15·m̂ (#48)."""
    import tm_bot.search.agent as search_agent

    pv = _make_tiny_policy_value_agent(seed=5)
    inner_predict = pv.predict_batch

    def predict_batch(*args: Any, **kwargs: Any) -> Any:
        out = inner_predict(*args, **kwargs)
        if kwargs.get("return_margin"):
            probs, values, margins = out
            return probs, values, [0.5] * len(margins)
        return out

    monkeypatch.setattr(pv, "predict_batch", predict_batch)
    seen: list[tuple[float, float]] = []
    real_utility = search_agent.search_utility

    def spy_utility(value: float, margin: float, *, weight: float) -> float:
        seen.append((margin, weight))
        return real_utility(value, margin, weight=weight)

    monkeypatch.setattr(search_agent, "search_utility", spy_utility)
    agent = ISMCTSAgent(pv, config=SearchConfig(num_simulations=4, num_worlds=2), seed=3)
    agent.reset(7)
    b0 = B0Agent(seed=1)
    with WorkerClient() as client:
        with client.open("search-margin-1", "tharsis") as live:
            while seats := live.pending():
                seat = seats[0]
                obs, choice = live.observe(seat), live.choice(seat)
                if seat == 0 and is_search_decision(obs, choice):
                    sampler = open_world_samplers(client, [(live.game_id, 0)])[0]
                    try:
                        agent.act_with_sampler(obs, choice, sampler)
                    finally:
                        release_samplers([sampler])
                    break
                live.choose(seat, b0.act(obs, choice))

    assert seen
    assert all(m == pytest.approx(0.5) and w == pytest.approx(0.15) for m, w in seen)


def test_search_config_simulations_worlds_determinism_and_deepcopy() -> None:
    pv = _make_tiny_policy_value_agent(seed=11)
    agent_fast = ISMCTSAgent(
        pv,
        config=SearchConfig(num_simulations=2, num_worlds=1),
        seed=55,
    )
    agent_more = ISMCTSAgent(
        pv,
        config=SearchConfig(num_simulations=6, num_worlds=3),
        seed=55,
    )
    assert agent_fast.config.num_simulations == 2
    assert agent_fast.config.num_worlds == 1
    assert agent_more.config.num_simulations == 6
    assert agent_more.config.num_worlds == 3

    # Deepcopy shares model/vocab weights while cloning per-game RNG & stats
    copied = copy.deepcopy(agent_more)
    assert copied.model is agent_more.model
    assert copied.vocab is agent_more.vocab
    assert copied.config == agent_more.config

    with pytest.raises(ValueError, match="action_temperature"):
        SearchConfig(action_temperature=-0.1)

    # Verify determinism across repeated runs with configured simulations, worlds, and the
    # v1 flat action_temperature (only live when play_temperature is None).
    with WorkerClient() as client:
        r1 = play_game(
            (ISMCTSAgent(pv, config=SearchConfig(num_simulations=4, num_worlds=2, action_temperature=1.0, play_temperature=None), seed=12), B1Agent(seed=34)),
            seed_key="eval-search-det-1",
            map_name="tharsis",
            client=client,
            seat_seeds=(100, 200),
        )
        r2 = play_game(
            (ISMCTSAgent(pv, config=SearchConfig(num_simulations=4, num_worlds=2, action_temperature=1.0, play_temperature=None), seed=12), B1Agent(seed=34)),
            seed_key="eval-search-det-1",
            map_name="tharsis",
            client=client,
            seat_seeds=(100, 200),
        )
        assert client.ping()["games"] == 0

    assert r1.ok is True
    assert r1.result == r2.result
    assert r1.decisions == r2.decisions
    assert r1.generations == r2.generations
    assert len(r1.searched_decision_latencies_ms[0]) > 0
    assert len(r1.searched_decision_latencies_ms[1]) == 0
    assert all(x >= 0.0 for x in r1.searched_decision_latencies_ms[0])


def test_search_runner_cleans_clones_on_exception_and_runs_batched_eval() -> None:
    class _FailingAfterSampleAgent:
        version = "fail-search-v1"

        def reset(self, seed: int | None = None) -> None:
            return None

        def act(self, observation: Any, choice: Any) -> int:
            return 0

        def act_with_sampler(
            self,
            observation: Any,
            choice: Any,
            sampler: WorldSampler,
        ) -> int:
            clones = sampler.sample(2, seed=7)
            assert len(clones) == 2
            raise RuntimeError("injected search crash")

    with WorkerPool(num_workers=2) as pool:
        outcomes = play_games(
            [
                GameSpec(
                    seed_key="search-crash-1",
                    map_name="tharsis",
                    agents=(_FailingAfterSampleAgent(), B0Agent(seed=1)),
                )
            ],
            client=pool,
            batch_size=1,
        )
        assert len(outcomes) == 1
        assert outcomes[0].ok is False
        assert outcomes[0].error_type == "RuntimeError"
        assert pool.active_games == 0

    # Per-game failure isolation across a batched search group: if one game errors
    # in act_batch_with_samplers, the healthy sibling game must still complete.
    class _OneBadGameBatchedSearchAgent:
        version = "batch-iso-v1"

        def reset(self, seed: int | None = None) -> None:
            return None

        def act(self, observation: Any, choice: Any) -> int:
            return 0

        def act_with_sampler(
            self,
            observation: Any,
            choice: Any,
            sampler: WorldSampler,
        ) -> int:
            sampler.sample(1, seed=11)
            if observation["public"]["map"] == "elysium":
                raise RuntimeError("injected elysium failure")
            return 0

        @classmethod
        def act_batch_with_samplers(
            cls,
            agents: Sequence[_OneBadGameBatchedSearchAgent],
            observations: Sequence[Any],
            choices: Sequence[Any],
            samplers: Sequence[WorldSampler],
        ) -> list[int]:
            for s in samplers:
                s.sample(1, seed=11)
            for obs in observations:
                if obs["public"]["map"] == "elysium":
                    raise RuntimeError("injected elysium failure")
            return [0 for _ in agents]

    with WorkerPool(num_workers=2) as pool:
        iso_outcomes = play_games(
            [
                GameSpec(
                    seed_key="search-iso-good",
                    map_name="tharsis",
                    agents=(_OneBadGameBatchedSearchAgent(), B0Agent(seed=1)),
                ),
                GameSpec(
                    seed_key="search-iso-bad",
                    map_name="elysium",
                    agents=(_OneBadGameBatchedSearchAgent(), B0Agent(seed=1)),
                ),
            ],
            client=pool,
            batch_size=2,
        )
        assert len(iso_outcomes) == 2
        assert iso_outcomes[0].ok is True
        assert iso_outcomes[1].ok is False
        assert iso_outcomes[1].error_type == "RuntimeError"
        assert pool.active_games == 0

    pv = _make_tiny_policy_value_agent(seed=21)
    report = evaluate_agents(
        agent_a=lambda: ISMCTSAgent(
            pv,
            config=SearchConfig(num_simulations=4, num_worlds=2),
            seed=0,
        ),
        agent_b=lambda: B1Agent(seed=0),
        agent_a_name="search-tiny",
        agent_b_name="b1",
        namespace="eval",
        num_seeds=1,
        maps=("tharsis",),
        num_workers=2,
        batch_size=2,
    )
    det = report["deterministic"]
    assert det["total_games"] == 2
    assert det["completed_games"] == 2
    assert det["failure_count"] == 0
    lat_searched = report["runtime"]["latency_ms"].get("agent_a_searched")
    assert lat_searched is not None
    assert lat_searched["decisions"] > 0
    assert lat_searched["p50_ms"] >= 0.0
    assert lat_searched["p95_ms"] >= lat_searched["p50_ms"]

    md = format_markdown_report(report)
    assert "Searched Decision Latency" in md


def test_warmstart_checkpoint_search_smoke_if_present() -> None:
    ckpt = Path(r"C:\2026\tm_bot\runs\warmstart\best\checkpoint.pt")
    if not ckpt.exists():
        pytest.skip("Warm-start checkpoint not present on this machine")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = ISMCTSAgent.from_checkpoint(
        ckpt,
        device=device,
        config=SearchConfig(num_simulations=4, num_worlds=2),
    )
    with WorkerClient() as client:
        res = play_game(
            (agent, B1Agent(seed=1)),
            seed_key="dev-warmstart-smoke-0",
            map_name="tharsis",
            client=client,
            seat_seeds=(11, 22),
        )
        assert client.ping()["games"] == 0
    assert res.ok is True
    assert len(res.searched_decision_latencies_ms[0]) > 0


@pytest.mark.parametrize("root_rule", ["completed_q", "visits"])
def test_search_records_root_stats_of_the_last_searched_decision(root_rule: str) -> None:
    """`last_root_stats` holds the root priors, visits and value sums (#57 measurement hook);
    `last_decision_policy` is π′ under the completed-Q rule (#60) and N/ΣN under the visit rule."""
    from tm_bot.search.agent import completed_q_policy

    pv = _make_tiny_policy_value_agent(seed=11)
    agent = ISMCTSAgent(pv, config=SearchConfig(num_simulations=4, num_worlds=2, root_rule=root_rule), seed=5)
    agent.reset(3)
    assert agent.last_root_stats is None
    b0 = B0Agent(seed=1)
    with WorkerClient() as client:
        with client.open("search-root-stats-1", "tharsis") as live:
            while seats := live.pending():
                seat = seats[0]
                obs, choice = live.observe(seat), live.choice(seat)
                if seat == 0 and is_search_decision(obs, choice):
                    sampler = open_world_samplers(client, [(live.game_id, 0)])[0]
                    try:
                        idx = agent.act_with_sampler(obs, choice, sampler)
                    finally:
                        release_samplers([sampler])
                    break
                live.choose(seat, b0.act(obs, choice))

    assert agent.last_root_stats is not None
    priors, visits, value_sums = agent.last_root_stats
    n = len(choice.candidates)
    assert priors.shape == visits.shape == value_sums.shape == (n,)
    assert visits.sum() == pytest.approx(4.0)
    if root_rule == "visits":
        assert agent.last_decision_policy == pytest.approx(visits / visits.sum())
    else:
        expected = completed_q_policy(priors, visits, value_sums, beta=3.0)
        assert agent.last_decision_policy == pytest.approx(expected)
        assert agent.last_decision_policy.sum() == pytest.approx(1.0)
    assert 0 <= idx < n
    agent.reset(4)
    assert agent.last_root_stats is None
