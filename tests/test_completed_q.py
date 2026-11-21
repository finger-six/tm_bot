"""Completed-Q root policy π′ ∝ prior · exp(β·Q̂) (#60) and the visit-count rule it replaces."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from tm_bot.search.agent import _root_decision, _TreeNode, completed_q_policy, search_agent_version
from tm_bot.search.config import ROOT_RULES, SearchConfig


def _root(priors: list[float], visits: list[float], value_sums: list[float]) -> _TreeNode:
    n = len(priors)
    N = np.asarray(visits, dtype=np.float64)
    return _TreeNode(
        gid=None, seat=0, num_actions=n, priors=np.asarray(priors, dtype=np.float64), N=N,
        W=np.asarray(value_sums, dtype=np.float64), VL=np.zeros(n), total_N=float(N.sum()),
    )


def _expected(priors: list[float], q: list[float], beta: float = 3.0) -> np.ndarray:
    w = np.asarray(priors) * np.exp(beta * np.asarray(q))
    return w / w.sum()


def test_pi_prime_equals_the_prior_without_visits() -> None:
    priors = np.array([0.5, 0.3, 0.2])
    pi = completed_q_policy(priors, np.zeros(3), np.zeros(3), beta=3.0)
    np.testing.assert_allclose(pi, priors)


def test_pi_prime_favours_the_higher_q_child_at_equal_prior() -> None:
    priors = [0.4, 0.4, 0.2]
    pi = completed_q_policy(np.array(priors), np.array([2.0, 2.0, 1.0]), np.array([1.0, -1.0, 0.0]), beta=3.0)
    np.testing.assert_allclose(pi, _expected(priors, [0.5, -0.5, 0.0]))
    assert pi[0] > pi[1]
    assert pi[0] / pi[1] == pytest.approx(math.exp(3.0 * 1.0))
    assert pi.sum() == pytest.approx(1.0)


def test_unvisited_children_get_the_mean_visited_q() -> None:
    # Visited Q: 0.6 and -0.2 (W / N), so their plain mean 0.2 fills the two unvisited children.
    priors = [0.25, 0.25, 0.25, 0.25]
    pi = completed_q_policy(np.array(priors), np.array([1.0, 3.0, 0.0, 0.0]), np.array([0.6, -0.6, 0.0, 0.0]),
                            beta=3.0)
    np.testing.assert_allclose(pi, _expected(priors, [0.6, -0.2, 0.2, 0.2]))
    assert pi[2] == pytest.approx(pi[3])


def test_zero_prior_children_stay_at_zero() -> None:
    pi = completed_q_policy(np.array([0.0, 1.0]), np.array([1.0, 1.0]), np.array([1.0, -1.0]), beta=3.0)
    np.testing.assert_allclose(pi, [0.0, 1.0])


def test_tau_zero_takes_the_argmax_of_pi_prime_and_records_it() -> None:
    # The prior's favourite (index 1) loses to the higher-Q child 0.
    root = _root([0.3, 0.5, 0.2], [2.0, 2.0, 0.0], [1.0, -1.0, 0.0])
    chosen, policy = _root_decision(root, rule="completed_q", beta=3.0, temperature=0.0, rng=random.Random(0))
    np.testing.assert_allclose(policy, completed_q_policy(root.priors, root.N, root.W, beta=3.0))
    assert chosen == int(np.argmax(policy)) == 0


def test_tau_zero_breaks_pi_prime_ties_by_prior() -> None:
    # 0.2·exp(3·ln2/3) == 0.4·exp(0): children 0 and 1 tie on π′; the higher prior (1) wins.
    q0 = math.log(2.0) / 3.0
    root = _root([0.2, 0.4, 0.4], [1.0, 1.0, 1.0], [q0, 0.0, -1.0])
    chosen, policy = _root_decision(root, rule="completed_q", beta=3.0, temperature=0.0, rng=random.Random(0))
    assert policy[0] == pytest.approx(policy[1])
    assert chosen == 1


@pytest.mark.parametrize("tau", [1.0, 0.5])
def test_sampled_tau_draws_from_pi_prime_to_the_one_over_tau(tau: float) -> None:
    root = _root([0.5, 0.3, 0.2], [2.0, 1.0, 1.0], [0.4, 0.3, -0.5])
    pi = completed_q_policy(root.priors, root.N, root.W, beta=3.0)
    target = pi ** (1.0 / tau)
    target /= target.sum()
    rng = random.Random(7)
    draws = 20_000
    counts = np.zeros(3)
    for _ in range(draws):
        chosen, policy = _root_decision(root, rule="completed_q", beta=3.0, temperature=tau, rng=rng)
        counts[chosen] += 1
    np.testing.assert_allclose(policy, pi)  # the recorded target is π′ itself, not π′^(1/τ)
    np.testing.assert_allclose(counts / draws, target, atol=0.015)


@pytest.mark.parametrize("rule", ROOT_RULES)
@pytest.mark.parametrize("tau", [0.0, 1.0])
def test_a_root_without_visits_plays_the_prior_under_both_rules(rule: str, tau: float) -> None:
    root = _root([0.2, 0.7, 0.1], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    picks = []
    for seed in range(50):
        chosen, policy = _root_decision(root, rule=rule, beta=3.0, temperature=tau, rng=random.Random(seed))
        np.testing.assert_allclose(policy, root.priors)
        picks.append(chosen)
        # Same draw as the raw policy head's `_select_from_probs` path with the same RNG state.
        rng = random.Random(seed)
        expected = 1 if tau == 0.0 else int(rng.choices(range(3), weights=root.priors.tolist(), k=1)[0])
        assert chosen == expected


def test_visit_count_rule_reproduces_the_v1_root_choice() -> None:
    root = _root([0.5, 0.3, 0.2], [3.0, 3.0, 2.0], [0.3, 0.9, 1.5])
    chosen, policy = _root_decision(root, rule="visits", beta=3.0, temperature=0.0, rng=random.Random(0))
    np.testing.assert_allclose(policy, root.N / root.N.sum())
    assert chosen == 1  # most visits, the tie between 0 and 1 broken by mean Q
    for seed in range(20):
        chosen, _ = _root_decision(root, rule="visits", beta=3.0, temperature=0.5, rng=random.Random(seed))
        weights = (root.N ** 2.0) / (root.N ** 2.0).sum()
        assert chosen == random.Random(seed).choices(range(3), weights=weights.tolist(), k=1)[0]


def test_search_config_defaults_to_completed_q_and_keeps_visits_selectable() -> None:
    cfg = SearchConfig()
    assert (cfg.root_rule, cfg.completed_q_beta) == ("completed_q", 3.0)
    assert SearchConfig.v2(fallback=True).root_rule == "completed_q"
    assert SearchConfig(root_rule="visits").root_rule == "visits"
    with pytest.raises(ValueError, match="root_rule"):
        SearchConfig(root_rule="argmax")
    with pytest.raises(ValueError, match="completed_q_beta"):
        SearchConfig(completed_q_beta=-1.0)


def test_version_tags_the_completed_q_rule_and_keeps_v1_tags() -> None:
    v1 = SearchConfig(num_simulations=8, play_temperature=None, margin_weight=0.0, root_rule="visits")
    assert search_agent_version(v1, "warmstart-v2") == "ismcts-s8-k4-warmstart-v2"
    light = SearchConfig(num_simulations=8, play_temperature=None, margin_weight=0.0)
    assert search_agent_version(light, "warmstart-v3") == "ismcts-s8-k4-cq3-warmstart-v3"
    assert search_agent_version(SearchConfig(), "net").endswith("-u0.15-cq3-net")
