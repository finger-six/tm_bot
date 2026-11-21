"""#57 diagnosis driver: the v2 main warm start vs b1-v2 under search/value/sub-choice variants.

Measurement only: no gate threshold, default `SearchConfig` value, encoder or training data changes.
Every variant plays the gate's seat-swapped `dev-*` schedule (13 seeds x 4 maps x 2 seats = 104
games by default) with the gate's pairing, runner and summary code, and writes one JSON per variant
to `--work-dir` (keep it outside the repo).

    # one variant (light search as gated, plus a per-decision log)
    python tools/diag57.py run --work-dir %TEMP%/diag57 --name light
    python tools/diag57.py run --work-dir %TEMP%/diag57 --name raw --sims 0
    python tools/diag57.py run --work-dir %TEMP%/diag57 --name fpu_parent --fpu parent --fpu-reduction 0.2
    # fit the val temperature of the value head (gen >= 2, as the comparator does)
    python tools/diag57.py fit-t --work-dir %TEMP%/diag57
    # markdown table of every finished variant
    python tools/diag57.py table --work-dir %TEMP%/diag57

Variant knobs (defaults = the gate's LIGHT_SEARCH): `--sims`, `--worlds`, `--wave-size`,
`--root {visits,prior_q}` and `--root-tau` (action-phase root rule; other phases keep τ = 0),
`--sub-tau` (policy-head sub-choices), `--fpu {zero,parent}` and `--fpu-reduction`,
`--leaf {raw,tscale,zero}` and `--leaf-temp`, `--hybrid` (space and payment sub-choices from
b1-v2's own heuristic: a measurement, never training data, #36), `--opponent {b1,self,v1}`.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

import tm_bot.search.agent as search_agent  # noqa: E402
from tm_bot.agents.b1 import B1Agent  # noqa: E402
from tm_bot.agents.runner import DEFAULT_MAX_DECISIONS, play_games  # noqa: E402
from tm_bot.env.types import SUPPORTED_MAPS  # noqa: E402
from tm_bot.eval.harness import _serialize_outcome, resolve_agent_factory  # noqa: E402
from tm_bot.eval.pairing import build_paired_specs  # noqa: E402
from tm_bot.models.agent import PolicyValueAgent  # noqa: E402
from tm_bot.search.agent import ISMCTSAgent, make_warmstart_search_agent  # noqa: E402
from tm_bot.search.config import is_search_decision  # noqa: E402
from tm_bot.training.run_warmstart_gate import LIGHT_SEARCH, value_entries_with_outcomes  # noqa: E402
from tm_bot.training.warmstart_gate import PairingSpec, gate_seed_keys, summarize_pairing  # noqa: E402

CHECKPOINT = REPO / "runs" / "warmstart-v3" / "best"
V1_CHECKPOINT = REPO / "runs" / "warmstart" / "best"
SUB_KINDS = ("space", "payment")

# --- variant knobs that live outside the agent (one variant per process) ------------------------

_FPU: dict[str, Any] = {"mode": "zero", "reduction": 0.0}
_original_select_puct = search_agent._select_puct


def _select_puct_parent_fpu(node: Any, *, sign: float, c_puct: float, fpu_value: float) -> int:
    """PUCT with FPU = the node's mean visited Q (selector's view) minus a reduction."""
    visited = node.N > 0.0
    if visited.any():
        fpu_value = sign * float(node.W[visited].sum() / node.N[visited].sum()) - _FPU["reduction"]
    return _original_select_puct(node, sign=sign, c_puct=c_puct, fpu_value=fpu_value)


class LeafPolicyValueAgent(PolicyValueAgent):
    """Transforms the value only at search leaves (the `return_margin=True` calls)."""

    leaf_mode = "raw"  # raw | tscale | zero
    leaf_temp = 1.0

    def predict_batch(self, observations, choices, *, compute_value=True, return_margin=False):  # type: ignore[no-untyped-def]
        out = super().predict_batch(observations, choices, compute_value=compute_value, return_margin=return_margin)
        if not return_margin or self.leaf_mode == "raw":
            return out
        probs, values, margins = out
        if self.leaf_mode == "zero":
            values = np.zeros_like(values)
        else:
            p = np.clip((np.asarray(values, dtype=np.float64) + 1.0) / 2.0, 1e-6, 1 - 1e-6)
            logit = np.log(p / (1.0 - p)) / self.leaf_temp
            values = (2.0 / (1.0 + np.exp(-logit)) - 1.0).astype(np.float32)
        return probs, values, margins


# --- branch categories (b1-v2's root action menu, tm_bot/agents/b1.py:413-436) ------------------


def branch_category(label: str) -> str:
    lbl = label.lower()
    if lbl.startswith("take first action of"):
        return "first_action"
    if "claim a milestone" in lbl:
        return "milestone"
    if "convert" in lbl and "greenery" in lbl:
        return "greenery"
    if "convert" in lbl and "heat" in lbl:
        return "heat"
    if "play project card" in lbl:
        return "play_card"
    if "perform an action" in lbl:
        return "card_action"
    if "fund an award" in lbl:
        return "award"
    if "standard projects" in lbl:
        return "std_proj"
    if "pass for this generation" in lbl:
        return "pass"
    if "end turn" in lbl:
        return "end_turn"
    return "other"


def _sample(weights: np.ndarray, rng: random.Random) -> int:
    w = np.asarray(weights, dtype=np.float64)
    w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
    if w.sum() <= 0.0:
        return int(np.argmax(weights))
    return int(rng.choices(range(len(w)), weights=(w / w.sum()).tolist(), k=1)[0])


def _tempered(probs: np.ndarray, tau: float, rng: random.Random) -> int:
    p = np.asarray(probs, dtype=np.float64)
    if tau <= 0.0:
        return int(np.argmax(p))
    if tau != 1.0:
        logits = np.log(np.clip(p, 1e-12, None)) / tau
        p = np.exp(logits - logits.max())
    return _sample(p, rng)


class DiagAgent(ISMCTSAgent):
    """The gated search agent plus variant root/sub-choice rules and per-game logs (measurement)."""

    def __init__(self, inner: ISMCTSAgent, opts: dict[str, Any], log: dict[str, Any]) -> None:
        super().__init__(inner.policy_agent, config=inner.config, seed=inner._seed, version=inner.version)
        self.opts = opts
        self.log = log
        self.b1 = B1Agent(seed=0) if opts["hybrid"] else None

    def __deepcopy__(self, memo: dict[int, Any]) -> DiagAgent:
        clone = super().__deepcopy__(memo)
        clone.opts = self.opts
        clone.log = self.log
        clone.b1 = copy.deepcopy(self.b1, memo) if self.b1 is not None else None
        return clone

    def _root_choice(self, obs: Any, n: int, default: int) -> int:
        """Re-choose an action-phase search root under the variant's root rule."""
        stats = self.last_root_stats
        if stats is None or str(obs.get("public", {}).get("phase", "")) != "action":
            return default
        priors, visits, wsum = stats
        rule, tau = self.opts["root"], self.opts["root_tau"]
        if rule == "visits" and tau is None:
            return default
        tau = 1.0 if tau is None else tau
        if visits.sum() <= 0.0:  # raw policy (0 sims) or failed sampling: the prior
            return _tempered(priors, tau, self._rng)
        if rule == "visits":
            if tau > 0.0:
                return _sample(np.power(visits, 1.0 / tau), self._rng)
            best = visits.max()
            tied = np.flatnonzero(visits == best).tolist()
            return int(max(tied, key=lambda a: (wsum[a] / max(1.0, visits[a]), priors[a], -a)))
        # prior_q: π' ∝ prior · exp(β·Q); unvisited children get the visited mean Q (no signal).
        q_vis = wsum[visits > 0] / visits[visits > 0]
        q = np.where(visits > 0, wsum / np.maximum(visits, 1.0), float(q_vis.mean()))
        logits = np.log(np.clip(priors, 1e-12, None)) + self.opts["beta"] * q
        return _tempered(np.exp(logits - logits.max()), tau, self._rng)

    @classmethod
    def act_batch_with_samplers(cls, agents, observations, choices, samplers):  # type: ignore[no-untyped-def]
        out = super().act_batch_with_samplers(agents, observations, choices, samplers)
        multi = [i for i, c in enumerate(choices) if len(c.candidates) > 1]
        for i in multi:
            ag, obs, ch = agents[i], observations[i], choices[i]
            n = len(ch.candidates)
            gen = int(obs.get("public", {}).get("generation", 0))
            phase = str(obs.get("public", {}).get("phase", ""))
            if is_search_decision(obs, ch):
                out[i] = ag._root_choice(obs, n, out[i])
                priors, visits, wsum = ag.last_root_stats
                if phase == "action":
                    prior_top = int(np.argmax(priors))
                    q = lambda a: float(wsum[a] / visits[a]) if visits[a] > 0 else None  # noqa: E731
                    ag.log["decisions"].append({
                        "gen": gen, "n": n, "chosen": int(out[i]), "prior_top": prior_top,
                        "cat": branch_category(ch.candidates[out[i]].label),
                        "prior_cat": branch_category(ch.candidates[prior_top].label),
                        "p_chosen": float(priors[out[i]]), "p_top": float(priors[prior_top]),
                        "visited": int((visits > 0).sum()), "max_n": float(visits.max()),
                        "q_chosen": q(out[i]), "q_top": q(prior_top),
                    })
            elif ag.b1 is not None and ch.kind in SUB_KINDS:
                out[i] = ag.b1.act(obs, ch)
            elif ag.opts["sub_tau"] is not None and ag.last_decision_policy is not None:
                out[i] = _tempered(ag.last_decision_policy, ag.opts["sub_tau"], ag._rng)
            ag.log["last_obs"] = obs
        if multi:  # the gate's live value diagnostic (ValueRecordingSearchAgent)
            _, values = agents[0].policy_agent.predict_batch(
                [observations[i] for i in multi], [choices[i] for i in multi], compute_value=True
            )
            for i, v in zip(multi, values):
                gen = int(observations[i].get("public", {}).get("generation", 0))
                agents[i].log["values"].append((gen, (float(v) + 1.0) / 2.0))
        return out


# --- VP breakdown from the candidate's last observation (GameResult has scores only) ------------


def vp_breakdown(obs: dict[str, Any], score: int, seat: int) -> dict[str, int]:
    pub = obs["public"]
    spaces = [sp for sp in pub["spaces"] if sp.get("spaceType") != "colony"]
    max_y = max(int(sp["y"]) for sp in spaces)
    by_coord = {(int(sp["x"]), int(sp["y"])): sp for sp in spaces}
    from tm_bot.agents.b1 import _adjacent_coords

    greenery = sum(1 for sp in spaces if sp["tileType"] == 0 and sp["owner"] == seat)
    cities = 0
    for sp in spaces:
        if sp["tileType"] in (2, 3) and sp["owner"] == seat:
            for c in _adjacent_coords(int(sp["x"]), int(sp["y"]), max_y):
                nb = by_coord.get(c)
                cities += int(nb is not None and nb["tileType"] == 0)
    milestones = 5 * sum(1 for m in pub["milestones"] if m["claimedBy"] == seat)
    awards = 0
    for a in pub["awards"]:
        if a["fundedBy"] is None:
            continue
        sc = {int(e["seat"]): int(e["score"]) for e in a["scores"]}
        if sc.get(seat, 0) >= max(sc.values(), default=0) and sc.get(seat, 0) > 0:
            awards += 5
    tr = int(pub["players"][seat]["terraformRating"])
    rest = tr + greenery + cities + milestones + awards
    return {"tr": tr, "greenery": greenery, "cities": cities, "milestones": milestones,
            "awards": awards, "cards_other": int(score) - rest, "total": int(score)}


# --- run ----------------------------------------------------------------------------------------


def _settings(args: argparse.Namespace) -> dict[str, Any]:
    return {**LIGHT_SEARCH, "num_simulations": args.sims, "num_worlds": args.worlds, "wave_size": args.wave_size}


def cmd_run(args: argparse.Namespace) -> int:
    import torch

    out_path = args.work_dir / f"{args.name}.json"
    if out_path.exists() and not args.force:
        print(f"[diag57] {out_path} exists; pass --force to rerun", flush=True)
        return 0
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.fpu == "parent":
        _FPU["reduction"] = args.fpu_reduction
        search_agent._select_puct = _select_puct_parent_fpu
    LeafPolicyValueAgent.leaf_mode = args.leaf
    LeafPolicyValueAgent.leaf_temp = args.leaf_temp
    settings = _settings(args)
    opts = {"root": args.root, "root_tau": args.root_tau, "beta": args.beta, "sub_tau": args.sub_tau,
            "hybrid": args.hybrid}
    base = make_warmstart_search_agent(checkpoint=str(args.checkpoint), device=device, **settings)
    if args.leaf != "raw":
        pv = copy.copy(base.policy_agent)
        pv.__class__ = LeafPolicyValueAgent
        base = ISMCTSAgent(pv, config=base.config, seed=base._seed, version=base.version)

    logs: list[dict[str, Any]] = []

    def candidate() -> DiagAgent:
        log: dict[str, Any] = {"values": [], "decisions": [], "last_obs": None}
        logs.append(log)
        return DiagAgent(base, opts, log)

    opp_settings = {**LIGHT_SEARCH, "num_simulations": args.opp_sims}
    opponents: dict[str, Callable[[], Any]] = {
        "b1": resolve_agent_factory("b1"),
        "self": lambda: make_warmstart_search_agent(checkpoint=str(args.checkpoint), device=device, **opp_settings),
        "v1": lambda: make_warmstart_search_agent(checkpoint=str(V1_CHECKPOINT), device=device, **opp_settings),
    }
    maps = tuple(args.maps) if args.maps else SUPPORTED_MAPS
    specs, metas = build_paired_specs(
        agent_a=candidate, agent_b=opponents[args.opponent], seed_keys=gate_seed_keys(args.seeds), maps=maps
    )
    print(f"[diag57] {args.name}: {len(specs)} games vs {args.opponent} ...", flush=True)
    t0 = time.perf_counter()
    outcomes = play_games(specs, num_workers=args.workers, batch_size=args.batch_size, max_decisions=DEFAULT_MAX_DECISIONS)
    wall = time.perf_counter() - t0
    values = value_entries_with_outcomes(outcomes, metas, [lg["values"] for lg in logs])
    spec = PairingSpec(f"diag_{args.name}", args.name, args.opponent, args.seeds, gating=False)
    summary = summarize_pairing(spec, outcomes, metas, candidate_name=args.name, opponent_name=args.opponent,
                                value_entries=values, wall_time_s=wall)
    games = []
    for out, meta, lg in zip(outcomes, metas, logs):
        g: dict[str, Any] = {"ok": out.ok, "a_seat": meta.a_seat, "map": meta.map_name, "seed": meta.seed_key}
        if out.ok and out.result is not None:
            res = out.result
            a = meta.a_seat
            g.update(score_a=int(res.scores[a]), score_b=int(res.scores[1 - a]),
                     won=0.5 if res.tie else float(res.winner == a), gens=int(res.generations))
            if lg["last_obs"] is not None:
                try:
                    g["vp_a"] = vp_breakdown(lg["last_obs"], res.scores[a], a)
                    g["vp_b"] = vp_breakdown(lg["last_obs"], res.scores[1 - a], 1 - a)
                except (KeyError, TypeError, ValueError) as exc:
                    g["vp_error"] = repr(exc)
        g["decisions"] = lg["decisions"]
        g["values"] = lg["values"]
        games.append(g)
    payload = {
        "name": args.name, "argv": sys.argv[1:], "settings": settings, "opts": opts,
        "fpu": dict(_FPU, mode=args.fpu), "leaf": {"mode": args.leaf, "temp": args.leaf_temp},
        "opponent": args.opponent, "opp_sims": args.opp_sims, "wall_time_s": wall,
        "summary": summary, "outcomes": [_serialize_outcome(o) for o in outcomes],
        "metas": [asdict(m) for m in metas], "games": games,
    }
    args.work_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload), encoding="utf-8")
    det = summary["report"]
    lv = summary["live_value"]["gen_2_plus"]
    print(f"[diag57] {args.name}: win {det['win_rate']:.3f} {det['win_rate_ci_95']} own VP {det['mean_vp_a']:.1f} "
          f"margin {det['mean_vp_margin']:+.1f} ECE(g>=2) {lv['ece']} acc {lv['accuracy']} wall {wall:.0f}s", flush=True)
    return 0


# --- fit T --------------------------------------------------------------------------------------


def cmd_fit_t(args: argparse.Namespace) -> int:
    import torch

    from tm_bot.comparator.calibration import fit_midgame_temperature
    from tm_bot.models.network import load_checkpoint
    from tm_bot.training.multitask import collect_value_outputs, load_warmstart_data

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, _, _ = load_checkpoint(args.checkpoint, map_location=device, encoder_version=None)
    model.to(device).eval()
    data = load_warmstart_data()
    values = collect_value_outputs(model, data, "val", device=device)
    fit = fit_midgame_temperature(values["logit"], values["won"], values["generation"])
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "temperature.json").write_text(json.dumps(fit, indent=2), encoding="utf-8")
    print(json.dumps(fit, indent=2), flush=True)
    return 0


# --- table --------------------------------------------------------------------------------------


def _ece(rows: list[tuple[float, float]], temp: float = 1.0) -> tuple[float | None, float | None, int]:
    from tm_bot.comparator.calibration import calibration_metrics

    if not rows:
        return None, None, 0
    p = np.clip(np.array([r[0] for r in rows]), 1e-6, 1 - 1e-6)
    if temp != 1.0:
        p = 1.0 / (1.0 + np.exp(-np.log(p / (1 - p)) / temp))
    m = calibration_metrics(p.tolist(), [r[1] for r in rows])
    return m["ece"], m["accuracy"], m["n"]


def cmd_table(args: argparse.Namespace) -> int:
    temp_path = args.work_dir / "temperature.json"
    temp = json.loads(temp_path.read_text())["temperature"] if temp_path.exists() else 1.0
    print(f"T (val, gen>=2) = {temp:.3f}\n")
    print("| variant | opp | n | win % [95% CI] | own VP | margin | ECE g>=2 (raw / T) | acc g>=2 | wall s |")
    print("|---|---|---|---|---|---|---|---|---|")
    files = sorted(args.work_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    for path in files:
        if path.name == "temperature.json":
            continue
        d = json.loads(path.read_text())
        det = d["summary"]["report"]
        ci = det["win_rate_ci_95"] or [float("nan")] * 2
        rows = [(p, u) for g in d["games"] if g.get("ok") and "won" in g for (gen, p) in g["values"]
                if gen >= 2 for u in [g["won"]]]
        e1, acc, n = _ece(rows)
        e2, _, _ = _ece(rows, temp)
        fmt = lambda x: "n/a" if x is None else f"{x:.3f}"  # noqa: E731
        print(f"| {d['name']} | {d['opponent']} | {det['completed_games']} | {100 * det['win_rate']:.1f} "
              f"[{100 * ci[0]:.1f}, {100 * ci[1]:.1f}] | {det['mean_vp_a']:.1f} | {det['mean_vp_margin']:+.1f} | "
              f"{fmt(e1)} / {fmt(e2)} | {fmt(acc)} | {d['wall_time_s']:.0f} |")
    return 0


def _pearson(x: list[float], y: list[float]) -> float:
    """Plain-Python Pearson r (np.corrcoef's BLAS call crashes natively in this env next to torch)."""
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def cmd_analyze(args: argparse.Namespace) -> int:
    """Root-visit spread, overrides by branch category, VP breakdown and calibration by generation."""
    for name in args.names:
        d = json.loads((args.work_dir / f"{name}.json").read_text())
        games = [g for g in d["games"] if g.get("ok") and "won" in g]
        decs = [(x, g) for g in games for x in g["decisions"]]
        print(f"\n## {name} ({len(games)} games, {len(decs)} action-phase root decisions)\n")
        if decs:
            searched = [x for x, _ in decs if x["max_n"] > 0]
            if searched:
                print(f"- children visited / legal: {np.mean([x['visited'] for x in searched]):.2f} / "
                      f"{np.mean([x['n'] for x in searched]):.2f}; max N = 1 (all ties) in "
                      f"{100 * np.mean([x['max_n'] <= 1 for x in searched]):.1f}% of roots")
            ov = [x["chosen"] != x["prior_top"] for x, _ in decs]
            print(f"- override rate (chosen != prior argmax): {100 * np.mean(ov):.1f}%; mean prior of chosen "
                  f"{np.mean([x['p_chosen'] for x, _ in decs]):.3f} vs prior argmax {np.mean([x['p_top'] for x, _ in decs]):.3f}")
            per_game = [(np.mean([x["chosen"] != x["prior_top"] for x in g["decisions"]]), g["won"],
                         g["score_a"] - g["score_b"]) for g in games if g["decisions"]]
            r = np.array(per_game)
            print(f"- per-game override rate vs margin: r = {_pearson(r[:, 0].tolist(), r[:, 2].tolist()):+.2f}; "
                  f"won games {100 * r[r[:, 1] == 1, 0].mean():.1f}% vs lost {100 * r[r[:, 1] == 0, 0].mean():.1f}%")
            print("\n| prior-argmax category | roots | override % | chosen share % (this cat) | pass chosen % |")
            print("|---|---|---|---|---|")
            cats = sorted({x["prior_cat"] for x, _ in decs} | {x["cat"] for x, _ in decs})
            for c in cats:
                rows = [x for x, _ in decs if x["prior_cat"] == c]
                chosen_share = 100 * np.mean([x["cat"] == c for x, _ in decs])
                if not rows:
                    print(f"| {c} | 0 | - | {chosen_share:.1f} | - |")
                    continue
                print(f"| {c} | {len(rows)} | {100 * np.mean([x['chosen'] != x['prior_top'] for x in rows]):.1f} | "
                      f"{chosen_share:.1f} | {100 * np.mean([x['cat'] == 'pass' for x in rows]):.1f} |")
        vp = [g for g in games if "vp_a" in g]
        if vp:
            keys = ("tr", "greenery", "cities", "milestones", "awards", "cards_other", "total")
            print("\n| VP (approx., last obs) | " + " | ".join(keys) + " |")
            print("|---|" + "---|" * len(keys))
            for side in ("vp_a", "vp_b"):
                print(f"| {name if side == 'vp_a' else d['opponent']} | "
                      + " | ".join(f"{np.mean([g[side][k] for g in vp]):.1f}" for k in keys) + " |")
        rows = [(gen, p, g["won"]) for g in games for (gen, p) in g["values"]]
        print("\n| gen bucket | n | mean P(win) | win rate |")
        print("|---|---|---|---|")
        for lo, hi, lab in ((1, 1, "gen 1"), (2, 5, "gen 2-5"), (6, 9, "gen 6-9"), (10, 99, "gen 10+")):
            b = [(p, u) for gen, p, u in rows if lo <= gen <= hi]
            if b:
                print(f"| {lab} | {len(b)} | {np.mean([p for p, _ in b]):.3f} | {np.mean([u for _, u in b]):.3f} |")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("run", "fit-t", "table", "analyze"):
        p = sub.add_parser(name)
        p.add_argument("--work-dir", type=Path, required=True, help="Scratch output dir (outside the repo).")
        p.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
        p.add_argument("--device", default=None)
        if name == "analyze":
            p.add_argument("names", nargs="+", help="Variant names (<work-dir>/<name>.json).")
        if name == "run":
            p.add_argument("--name", required=True)
            p.add_argument("--force", action="store_true")
            p.add_argument("--sims", type=int, default=LIGHT_SEARCH["num_simulations"])
            p.add_argument("--worlds", type=int, default=LIGHT_SEARCH["num_worlds"])
            p.add_argument("--wave-size", type=int, default=LIGHT_SEARCH["wave_size"])
            p.add_argument("--root", choices=("visits", "prior_q"), default="visits")
            p.add_argument("--root-tau", type=float, default=None, help="Action-phase root τ (default: config's 1.0).")
            p.add_argument("--beta", type=float, default=3.0, help="prior_q: weight of Q on the prior's logits.")
            p.add_argument("--sub-tau", type=float, default=None, help="Policy sub-choice τ (default: config's 1.0).")
            p.add_argument("--fpu", choices=("zero", "parent"), default="zero")
            p.add_argument("--fpu-reduction", type=float, default=0.0)
            p.add_argument("--leaf", choices=("raw", "tscale", "zero"), default="raw")
            p.add_argument("--leaf-temp", type=float, default=1.0)
            p.add_argument("--hybrid", action="store_true", help="space/payment sub-choices from b1-v2 (measurement).")
            p.add_argument("--opponent", choices=("b1", "self", "v1"), default="b1")
            p.add_argument("--opp-sims", type=int, default=0, help="Simulations of a self/v1 opponent.")
            p.add_argument("--seeds", type=int, default=13)
            p.add_argument("--maps", nargs="+", default=None)
            p.add_argument("--workers", type=int, default=8)
            p.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    if args.work_dir.resolve().is_relative_to(REPO.resolve()):
        parser.error("--work-dir must be outside the repo")
    return {"run": cmd_run, "fit-t": cmd_fit_t, "table": cmd_table, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
