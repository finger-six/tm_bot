"""Generation 2+ log-state reconstruction for the v2 warm-start tables (one perspective, one pass).

`reconstruct_log_state` walks one perspective file's `moves` once and emits `LogDecision`s for both
seats: gen 2+ draft picks, research buys, deal-boundary value points, and strided action-phase value
points. Each decision carries the shared snapshot column group (`Snapshot` fields, actor-relative)
and a payload (the label side). The snapshot is causal: a decision at move `k` reads only rows
`<= k-1` (a deal-boundary point reads the deal row itself, the first research-phase state), the
actor's own setup, draft, research-buy and tracked-hand history, and public tableaus. Opponent
private hands, draws, keeps and draft packets never enter a snapshot. Gate outcomes (`drop_reason`)
may look ahead; they select rows but never feed features.

Timing rules (archive audit 1.2): globals, TR, productions, tags, milestones and awards come from
row `k-1`'s `game_state`. Stocks are stale on the deal row by the production income and research-buy
payments do not post, so each seat's six stocks are reconstructed: a tracker value that changed on a
row is taken as posted; otherwise the row's parsed `NAME gains/pays/loses N <resource>` segments are
applied; the deal row applies the production phase (`M€ += M€ production + TR`, energy to heat, the
other stocks gain their production).

Income identity (Gate 1). At each gen 2+ deal row the true generation-start M€ is
`income_identity_mc(snapshot M€, M€ production, TR)`; the M€ at a later decision subtracts this
seat's `NAME pays N M€` segments since the deal. Cross-check rule `INCOME_CHECK_*`: walk the seat's
later rows (until the next deal row or `INCOME_CHECK_WINDOW_ROWS` rows) keeping
`expected = corrected - payments + parsed M€ gains/losses`; every row where the seat's tracker M€
changes is compared to `expected`. The first agreement passes the seat-generation; it fails when
`INCOME_CHECK_STRIKES` (2) such snapshots in a row disagree; a window that ends before either is
`unverified` (kept, counted). A failed seat-generation drops all of that seat's rows in that
generation (draft picks, research buy, value points).

Exact-hand tracking (Gate 4), per seat: setup `chosen_projects`, plus research-buy keeps, named
draws and named keeps, minus plays and named discards. The first unrecoverable identity loss (a
count-only discard that does not empty the hand, an unnamed draw, a play missing from the hand, a
missing or repeated research keep row) sets `hand_lost_at[pid]`; that seat's later policy rows and
action value points are dropped (forward quarantine), and deal-boundary value points keep only the
public hand count.

Research-buy gate (Gate 3): the packet is the seat's 3 verified picks plus the final 1-card pass
packet (4 distinct cards), `kept <= packet`, `paid == 3 * |kept|` from the row's own payment text,
and the reconstructed M€ covers the payment; discards are the set difference.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Collection

from tm_bot.data.cards import CardRegistry
from tm_bot.data.gates import DraftExtractionResult, SetupGateResult, VerifiedDraftPick
from tm_bot.data.inventory import _opt_str

MC = "M€"
STOCK_KEYS: tuple[tuple[str, str], ...] = (
    ("mc", MC), ("steel", "Steel"), ("titanium", "Titanium"), ("plants", "Plant"),
    ("energy", "Energy"), ("heat", "Heat"),
)
PROD_KEYS: tuple[tuple[str, str], ...] = tuple((f"{k}_prod", f"{v} Production") for k, v in STOCK_KEYS)
# Order of the `me_tags` / `opp_tags` snapshot columns: the 11 tracker tag counts plus played events.
SNAPSHOT_TAGS: tuple[str, ...] = (
    "building", "space", "science", "power", "earth", "jovian", "plant", "microbe", "animal", "city",
    "wild", "event",
)
_TAG_TRACKERS: tuple[str, ...] = tuple(
    "Count of played Events cards" if t == "event" else f"Count of {t.capitalize()} tags" for t in SNAPSHOT_TAGS
)
_RESOURCE_TO_STOCK = {v: k for k, v in STOCK_KEYS}

CARD_COST = 3
INCOME_CHECK_WINDOW_ROWS = 60
INCOME_CHECK_STRIKES = 2

ROOT_ACTION_TYPES = frozenset(
    {"play_card", "standard_project", "claim_milestone", "fund_award", "activate_card", "convert_heat",
     "convert_plants"}
)
# The first row of one of these types after a deal row ends that generation's research window.
_ACTION_PHASE_TYPES = ROOT_ACTION_TYPES | {"place_tile", "draw"}

_AMOUNT_RE = re.compile(r"^(pays|gains|loses) (\d+) (M€|Steel|Titanium|Plant|Energy|Heat)(?!\w)")
_DRAWS_LIST_RE = re.compile(r"^draws \d+ cards?: (.+)$")
_DRAWS_COUNT_RE = re.compile(r"^draws (?:\d+ cards?|a card)\s*(?:\(.*\))?$")
_DRAWS_NAMED_RE = re.compile(r"^draws (.+?)\s*(?:\(.*\))?$")
_DISCARDS_COUNT_RE = re.compile(r"^discards (\d+|a) card(?:/s|s)?\b")
_DISCARDS_NAMED_RE = re.compile(r"^discards (.+?)\s*(?:\(.*\))?$")


def income_identity_mc(
    snapshot_mc: int, mc_production: int, terraform_rating: int, payments: Collection[int] = ()
) -> int:
    """True M€ at a gen 2+ research/draft decision: `snapshot M€ + M€ production + TR - payments`.

    `snapshot_mc`, `mc_production` and `terraform_rating` are the seat's values on the deal row
    (stale by the production income); `payments` are the seat's `pays N M€` amounts since the deal.
    """
    return int(snapshot_mc) + int(mc_production) + int(terraform_rating) - sum(int(p) for p in payments)


@dataclass(frozen=True)
class LogDecision:
    kind: str  # "draft_pick" | "research_buy" | "deal_boundary" | "action"
    player_id: str
    seat: int
    generation: int
    move_number: int
    snapshot: dict[str, Any]
    payload: dict[str, Any] = field(default_factory=dict)
    drop_reason: str | None = None


@dataclass
class LogStateResult:
    table_reason: str | None = None
    decisions: list[LogDecision] = field(default_factory=list)
    counts: Counter = field(default_factory=Counter)
    counts_by_player: dict[str, Counter] = field(default_factory=dict)
    gen_start_mc: dict[tuple[str, int], int] = field(default_factory=dict)
    income_checks: dict[tuple[str, int], str] = field(default_factory=dict)
    hand_lost_at: dict[str, int] = field(default_factory=dict)

    def count(self, pid: str, key: str) -> None:
        """Count one gate outcome, in total and for the seat it belongs to."""
        self.counts[key] += 1
        self.counts_by_player.setdefault(pid, Counter())[key] += 1


@dataclass
class _Seat:
    pid: str
    name: str
    seat: int
    corp: str
    hand: set[str]
    preludes_in_hand: set[str]
    setup: dict[str, list[str]]
    tableau: list[str] = field(default_factory=list)
    stocks: dict[str, int] = field(default_factory=lambda: {k: 0 for k, _ in STOCK_KEYS})
    hand_lost_at: int | None = None
    offer: set[str] = field(default_factory=set)
    research: list[dict[str, Any]] = field(default_factory=list)
    discarded_seen: list[str] = field(default_factory=list)


@dataclass
class _IncomeCheck:
    generation: int
    expected: int
    deal_index: int
    strikes: int = 0
    status: str | None = None


def _int(v: Any) -> int:
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0


def _segments(desc: Any) -> list[str]:
    return [s.strip() for s in desc.split("|")] if isinstance(desc, str) else []


def _split_names(text: str) -> list[str]:
    return [n.strip() for n in text.split(",") if n.strip()]


def _resolve_all(names: list[Any], registry: CardRegistry) -> list[str] | None:
    out = []
    for n in names:
        c = registry.resolve(n)
        if c is None:
            return None
        out.append(c)
    return out


def _is_setup_keep(st: _Seat, kept: list[Any], registry: CardRegistry) -> bool:
    return any(registry.resolve(c) == st.corp for c in kept)


def check_v2_file(moves: list[Any]) -> str | None:
    """Table-level v2 gates on one perspective: timing and degenerate-file styles (audit 1.1-1.2)."""
    if not any(isinstance(m, dict) and isinstance(m.get("game_state"), dict) for m in moves):
        return "missing_game_state"
    trackers_seen = False
    for idx, m in enumerate(moves):
        if not isinstance(m, dict):
            continue
        gs = m.get("game_state")
        if not isinstance(gs, dict) or gs.get("move_number") != m.get("move_number"):
            return "timing_mismatch"
        if gs.get("player_trackers"):
            trackers_seen = True
        elif idx > 0:
            return "degenerate_empty_trackers"
        act = _opt_str(m.get("action_type"))
        desc = m.get("description") if isinstance(m.get("description"), str) else ""
        if act in ("draft", "draft_card") and m.get("card_drafted") is None and not re.search(r"drafts \d+ cards?:", desc):
            return "degenerate_unstructured_draft"
    return None if trackers_seen else "degenerate_empty_trackers"


def strided_action_moves(moves: list[Any], player_ids: Collection[str]) -> set[int]:
    """Deterministic action snapshot positions: each seat's root action rows at ~1/3 and ~2/3 per gen 2+."""
    rows: dict[tuple[str, int], list[int]] = {}
    for m in moves:
        if not isinstance(m, dict) or _opt_str(m.get("action_type")) not in ROOT_ACTION_TYPES:
            continue
        pid = _opt_str(m.get("player_id"))
        gs = m.get("game_state")
        gen = _int(gs.get("generation")) if isinstance(gs, dict) else 0
        if pid in player_ids and gen >= 2:
            rows.setdefault((pid, gen), []).append(_int(m.get("move_number")))
    out: set[int] = set()
    for moves_in_gen in rows.values():
        n = len(moves_in_gen)
        out.update({moves_in_gen[n // 3], moves_in_gen[(2 * n) // 3]})
    return out


def reconstruct_log_state(
    data: dict[str, Any],
    *,
    setup: SetupGateResult,
    drafts: DraftExtractionResult,
    card_registry: CardRegistry,
    action_moves: Collection[int] | None = None,
) -> LogStateResult:
    """Single pass over one perspective's moves; see the module docstring for the rules."""
    result = LogStateResult()
    moves = data.get("moves")
    if not isinstance(moves, list) or not setup.passed:
        result.table_reason = "missing_moves" if not isinstance(moves, list) else "setup_failed"
        return result
    reason = check_v2_file(moves)
    if reason is not None:
        result.table_reason = reason
        return result

    seats: dict[str, _Seat] = {}
    for ps in setup.players:
        seats[ps.player_id] = _Seat(
            pid=ps.player_id,
            name=ps.player_name or ps.player_id,
            seat=ps.seat,
            corp=ps.chosen_corporation,
            hand=set(ps.chosen_projects),
            preludes_in_hand=set(ps.chosen_preludes),
            setup={
                "corps_discarded": sorted(c for c in ps.offered_corporations if c != ps.chosen_corporation),
                "preludes": sorted(ps.chosen_preludes),
                "preludes_discarded": sorted(set(ps.offered_preludes) - set(ps.chosen_preludes)),
                "projects_bought": sorted(ps.chosen_projects),
                "projects_discarded": sorted(set(ps.offered_projects) - set(ps.chosen_projects)),
            },
            tableau=[ps.chosen_corporation],
        )
    names = sorted(((s.name, s.pid) for s in seats.values()), key=lambda t: -len(t[0]))
    if action_moves is None:
        action_moves = strided_action_moves(moves, seats.keys())

    picks_by_move: dict[int, VerifiedDraftPick] = {p.move_number: p for p in drafts.picks}
    picks_by_gen: dict[tuple[str, int], list[VerifiedDraftPick]] = {}
    for p in drafts.picks:
        picks_by_gen.setdefault((p.player_id, p.generation), []).append(p)

    prev_gs: dict[str, Any] | None = None
    gen = 1
    research_open = False
    last_packet: dict[str, list[str]] = {}
    drafted_round: dict[str, list[str]] = {pid: [] for pid in seats}
    kept_this_gen: dict[str, list[str] | None] = {pid: None for pid in seats}
    checks: dict[str, _IncomeCheck] = {}

    def lose_hand(st: _Seat, move: int, why: str) -> None:
        if st.hand_lost_at is None:
            st.hand_lost_at = move
            result.hand_lost_at[st.pid] = move
            result.count(st.pid, f"hand_lost:{why}")

    def split_actor(seg: str) -> tuple[_Seat | None, str]:
        for nm, pid in names:
            if seg.startswith(nm + " "):
                return seats[pid], seg[len(nm) + 1:]
        return None, seg

    def snapshot(actor: _Seat, gs: dict[str, Any], phase: str) -> dict[str, Any]:
        opp = next(s for s in seats.values() if s.pid != actor.pid)
        trackers = gs.get("player_trackers") or {}
        vp = gs.get("player_vp") or {}
        snap: dict[str, Any] = {
            "phase": phase,
            "temperature": _int(gs.get("temperature")),
            "oxygen": _int(gs.get("oxygen")),
            "oceans": _int(gs.get("oceans")),
        }
        for prefix, st in (("me", actor), ("opp", opp)):
            tr = trackers.get(st.pid) or {}
            pvp = vp.get(st.pid) or {}
            details = pvp.get("details") or {}
            snap[f"{prefix}_tr"] = _int((pvp.get("total_details") or {}).get("tr"))
            for key, _ in STOCK_KEYS:
                snap[f"{prefix}_{key}"] = st.stocks[key]
            for key, tkey in PROD_KEYS:
                snap[f"{prefix}_{key}"] = _int(tr.get(tkey))
            snap[f"{prefix}_tags"] = [_int(tr.get(t)) for t in _TAG_TRACKERS]
            snap[f"{prefix}_cities"] = len(details.get("cities") or {})
            snap[f"{prefix}_greeneries"] = _int((pvp.get("total_details") or {}).get("greeneries"))
            snap[f"{prefix}_hand_count"] = _int(tr.get("Hand Counter"))
            snap[f"{prefix}_tableau"] = sorted(st.tableau)
        exact = actor.hand_lost_at is None
        snap["me_hand_exact"] = exact
        snap["me_hand"] = sorted(actor.hand) if exact else []
        if exact:
            snap["me_hand_count"] = len(actor.hand) + len(actor.preludes_in_hand)
        snap["me_preludes_in_hand"] = sorted(actor.preludes_in_hand)
        snap["me_drafted_this_round"] = list(drafted_round[actor.pid]) if phase == "drafting" else []
        for kind, key in (("milestone", "milestones"), ("award", "awards")):
            entries = sorted((k, v) for k, v in (gs.get(key) or {}).items() if isinstance(v, dict))
            snap[f"{kind}_names"] = [k for k, _ in entries]
            snap[f"{kind}_seats"] = [seats[v["player_id"]].seat if v.get("player_id") in seats else -1 for _, v in entries]
        snap["me_corporation"] = actor.corp
        snap["me_setup_preludes"] = actor.setup["preludes"]
        snap["me_setup_projects_bought"] = actor.setup["projects_bought"]
        snap["me_setup_corps_discarded"] = actor.setup["corps_discarded"]
        snap["me_setup_preludes_discarded"] = actor.setup["preludes_discarded"]
        snap["me_setup_projects_discarded"] = actor.setup["projects_discarded"]
        snap["me_research_generations"] = [r["generation"] for r in actor.research]
        snap["me_research_offered"] = [list(r["offered"]) for r in actor.research]
        snap["me_research_kept"] = [list(r["kept"]) for r in actor.research]
        snap["me_research_discarded"] = [list(r["discarded"]) for r in actor.research]
        snap["me_discarded_seen"] = sorted(actor.discarded_seen)
        return snap

    def emit(kind: str, st: _Seat, move: int, gs: dict[str, Any], phase: str, payload: dict[str, Any],
             drop: str | None = None) -> None:
        result.decisions.append(
            LogDecision(kind=kind, player_id=st.pid, seat=st.seat, generation=gen, move_number=move,
                        snapshot=snapshot(st, gs, phase), payload=payload, drop_reason=drop)
        )

    def end_research_window(move: int) -> None:
        nonlocal research_open
        research_open = False
        for pid, st in seats.items():
            if kept_this_gen[pid] is None and len(picks_by_gen.get((pid, gen), [])) > 0:
                result.count(pid, "research_buy:missing_keep_row")
                lose_hand(st, move, "missing_keep_row")

    def research_keep(st: _Seat, m: dict[str, Any], move: int, kept_raw: list[Any], segs: list[str],
                      gs_prev: dict[str, Any]) -> None:
        """Gate 3 on one research keep row, observed before the row's payment applies."""
        kept = _resolve_all(kept_raw, card_registry)
        if kept is None:
            result.count(st.pid, "research_buy:unresolvable_card_name")
            lose_hand(st, move, "unresolvable_keep")
            return
        prior = kept_this_gen[st.pid]
        paid = []
        for seg in segs:
            who, rest = split_actor(seg)
            mt = _AMOUNT_RE.match(rest) if who is st else None
            if mt and mt.group(1) == "pays" and mt.group(3) == MC:
                paid.append(int(mt.group(2)))
        if prior is not None:
            if sorted(kept) == sorted(prior) and not paid:
                return  # echo of the verified keep row
            result.count(st.pid, "research_buy:repeated_keep_row")
            lose_hand(st, move, "repeated_keep_row")
            return
        kept_this_gen[st.pid] = kept
        picks = sorted(picks_by_gen.get((st.pid, gen), []), key=lambda p: p.pick_index)
        final = last_packet.get(st.pid) or []
        packet = [p.chosen_card for p in picks] + (final if len(final) == 1 else [])
        drop = None
        if [p.pick_index for p in picks] != [0, 1, 2] or len(final) != 1 or len(set(packet)) != 4:
            drop = "incomplete_packet"
        elif not set(kept) <= set(packet):
            drop = "kept_not_in_packet"
        elif len(paid) != 1:
            drop = "missing_payment_text"
        elif paid[0] != CARD_COST * len(kept):
            drop = "payment_mismatch"
        elif st.stocks["mc"] < paid[0]:
            drop = "unaffordable"
        if drop is None and st.hand_lost_at is not None:
            drop = "hand_lost"
        if drop is not None:
            result.count(st.pid, f"research_buy:{drop}")
        payload = {"packet_cards": sorted(packet), "kept_cards": sorted(kept), "paid_mc": paid[0] if paid else None}
        emit("research_buy", st, move, gs_prev, "research", payload, drop)
        st.hand.update(kept)
        if drop is None:
            st.research.append({"generation": gen, "offered": sorted(packet), "kept": sorted(kept),
                                "discarded": sorted(set(packet) - set(kept))})

    for idx, m in enumerate(moves):
        if not isinstance(m, dict):
            continue
        gs = m["game_state"]
        move = _int(m.get("move_number"))
        pid = _opt_str(m.get("player_id"))
        act = _opt_str(m.get("action_type"))
        segs = _segments(m.get("description"))
        row_gen = _int(gs.get("generation"))
        is_deal = prev_gs is not None and row_gen > gen and row_gen >= 2

        # ---- decisions observed before this row is applied (state = rows <= move - 1) ----------
        handled_keep = False
        if prev_gs is not None and gen >= 2 and not is_deal:
            if research_open and act in _ACTION_PHASE_TYPES:
                end_research_window(move)
            pick = picks_by_move.get(move)
            if pick is not None and pick.player_id in seats:
                st = seats[pick.player_id]
                drop = None
                if pick.generation != gen:
                    drop = "draft_generation_mismatch"
                elif st.hand_lost_at is not None:
                    drop = "hand_lost"
                if drop is not None:
                    result.count(st.pid, f"draft_pick:{drop}")
                emit("draft_pick", st, move, prev_gs, "drafting",
                     {"draft_round_id": pick.draft_round_id, "pick_index": pick.pick_index,
                      "packet_cards": list(pick.packet_cards), "chosen_card": pick.chosen_card}, drop)
            keep_row = m.get("cards_kept") if isinstance(m.get("cards_kept"), dict) else {}
            own = keep_row.get(pid) if pid in seats else None
            if research_open and act != "draw" and isinstance(own, list) and own and not _is_setup_keep(seats[pid], own, card_registry):
                research_keep(seats[pid], m, move, own, segs, prev_gs)
                handled_keep = True
            if move in action_moves and pid in seats and act in ROOT_ACTION_TYPES and not research_open:
                st = seats[pid]
                drop = "hand_lost" if st.hand_lost_at is not None else None
                if drop is not None:
                    result.count(st.pid, "action:hand_lost")
                emit("action", st, move, prev_gs, "action", {}, drop)

        # ---- apply this row ---------------------------------------------------------------------
        trackers = gs.get("player_trackers") or {}
        prev_trackers = (prev_gs or {}).get("player_trackers") or {}
        deltas: dict[str, Counter] = {p: Counter() for p in seats}
        if not is_deal:
            for seg in segs:
                who, rest = split_actor(seg)
                mt = _AMOUNT_RE.match(rest) if who is not None else None
                if mt:
                    sign = 1 if mt.group(1) == "gains" else -1
                    deltas[who.pid][_RESOURCE_TO_STOCK[mt.group(3)]] += sign * int(mt.group(2))
        for p, st in seats.items():
            cur, prev = trackers.get(p) or {}, prev_trackers.get(p) or {}
            posted = {key: prev_gs is not None and cur.get(tk) != prev.get(tk) or prev_gs is None for key, tk in STOCK_KEYS}
            if is_deal:
                tr_val = _int(((gs.get("player_vp") or {}).get(p) or {}).get("total_details", {}).get("tr"))
                before_mc = st.stocks["mc"]
                prod = {key: _int(cur.get(tk)) for key, tk in PROD_KEYS}
                new = dict(st.stocks)
                new["mc"] = income_identity_mc(before_mc, prod["mc_prod"], tr_val)
                new["heat"] = st.stocks["heat"] + st.stocks["energy"] + prod["heat_prod"]
                new["energy"] = prod["energy_prod"]
                for key in ("steel", "titanium", "plants"):
                    new[key] = st.stocks[key] + prod[f"{key}_prod"]
                for key, tk in STOCK_KEYS:
                    st.stocks[key] = _int(cur.get(tk)) if posted[key] else new[key]
                result.gen_start_mc[(p, row_gen)] = st.stocks["mc"]
            else:
                for key, tk in STOCK_KEYS:
                    st.stocks[key] = _int(cur.get(tk)) if posted[key] else st.stocks[key] + deltas[p][key]
            # Gate 1 cross-check walk.
            chk = checks.get(p)
            if chk is not None and chk.status is None and not is_deal:
                chk.expected += deltas[p]["mc"]
                if cur.get(MC) != prev.get(MC):
                    if _int(cur.get(MC)) == chk.expected:
                        chk.status = "pass"
                    else:
                        chk.strikes += 1
                        if chk.strikes >= INCOME_CHECK_STRIKES:
                            chk.status = "fail"
                if chk.status is None and idx - chk.deal_index >= INCOME_CHECK_WINDOW_ROWS:
                    chk.status = "unverified"

        if is_deal:
            for p, chk in checks.items():
                result.income_checks[(p, chk.generation)] = chk.status or "unverified"
            if research_open:
                end_research_window(move)
            gen = row_gen
            research_open = True
            last_packet.clear()
            for p in seats:
                drafted_round[p] = []
                kept_this_gen[p] = None
                checks[p] = _IncomeCheck(generation=gen, expected=seats[p].stocks["mc"], deal_index=idx)

        opts = m.get("card_options")
        if gen >= 2 and research_open and isinstance(opts, dict) and act != "draw":
            for opid, cards in opts.items():
                resolved = _resolve_all(cards, card_registry) if isinstance(cards, list) else None
                if resolved is not None and str(opid) in seats:
                    last_packet[str(opid)] = resolved
        pick = picks_by_move.get(move)
        if pick is not None and pick.player_id in drafted_round:
            drafted_round[pick.player_id].append(pick.chosen_card)

        # Hand and tableau events, attributed by player_id (structured fields) or segment names.
        kept_map = m.get("cards_kept") if isinstance(m.get("cards_kept"), dict) else {}
        actor = seats.get(pid or "")
        own_kept = kept_map.get(pid) if actor is not None else None
        handled_draw = False
        if actor is not None and act == "draw":
            offer = opts.get(pid) if isinstance(opts, dict) else None
            if isinstance(own_kept, list) and own_kept:
                got = _resolve_all(own_kept, card_registry)
                if got is None:
                    lose_hand(actor, move, "unresolvable_draw")
                else:
                    actor.hand.update(got)
                handled_draw = True
            elif isinstance(offer, list) and offer:
                got = _resolve_all(offer, card_registry)
                actor.offer = set(got or [])
                handled_draw = True
        elif actor is not None and isinstance(own_kept, list) and own_kept and not handled_keep and not _is_setup_keep(
            actor, own_kept, card_registry
        ):
            got = _resolve_all(own_kept, card_registry)
            prior = kept_this_gen.get(actor.pid)
            if got is None:
                lose_hand(actor, move, "unresolvable_keep")
            elif prior is not None and sorted(got) == sorted(prior):
                pass  # echo of this generation's research keep
            else:
                actor.hand.update(got)
                if actor.offer:
                    actor.discarded_seen.extend(sorted(actor.offer - set(got)))
                    actor.offer = set()
        for seg in segs:
            who, rest = split_actor(seg)
            if who is None:
                continue
            if rest.startswith("plays card "):
                card = card_registry.resolve(rest[len("plays card "):].strip())
                if card is None:
                    lose_hand(who, move, "unresolvable_play")
                    continue
                who.tableau.append(card)
                if card in who.hand:
                    who.hand.discard(card)
                elif card in who.preludes_in_hand:
                    who.preludes_in_hand.discard(card)
                elif card in who.offer:
                    who.offer.discard(card)
                    who.discarded_seen.extend(sorted(who.offer))
                    who.offer = set()
                else:
                    lose_hand(who, move, "play_not_in_hand")
            elif rest.startswith("draws ") and not handled_draw:
                lst = _DRAWS_LIST_RE.match(rest)
                if lst is None and _DRAWS_COUNT_RE.match(rest):
                    lose_hand(who, move, "unnamed_draw")
                    continue
                named = _split_names(lst.group(1)) if lst else _split_names(_DRAWS_NAMED_RE.match(rest).group(1))
                got = _resolve_all(named, card_registry)
                if got is None:
                    lose_hand(who, move, "unnamed_draw")
                else:
                    who.hand.update(got)
            elif rest.startswith("discards "):
                cnt = _DISCARDS_COUNT_RE.match(rest)
                if cnt:
                    n = 1 if cnt.group(1) == "a" else int(cnt.group(1))
                    if n == len(who.hand):
                        who.hand.clear()
                    else:
                        lose_hand(who, move, "count_only_discard")
                    continue
                named = _resolve_all(_split_names(_DISCARDS_NAMED_RE.match(rest).group(1)), card_registry)
                if named is None or not set(named) <= who.hand:
                    lose_hand(who, move, "unmatched_discard")
                else:
                    who.hand.difference_update(named)
                    who.discarded_seen.extend(named)

        if is_deal:
            for st in seats.values():
                emit("deal_boundary", st, move, gs, "research", {})
        prev_gs = gs

    for p, chk in checks.items():
        result.income_checks[(p, chk.generation)] = chk.status or "unverified"
    _apply_income_quarantine(result)
    return result


def _apply_income_quarantine(result: LogStateResult) -> None:
    """Count Gate 1 outcomes and drop every row of a failed seat-generation."""
    for (pid, _), status in result.income_checks.items():
        result.count(pid, f"income_identity:{status}")
    out: list[LogDecision] = []
    for d in result.decisions:
        if result.income_checks.get((d.player_id, d.generation)) == "fail" and d.drop_reason is None:
            result.count(d.player_id, f"{d.kind}:income_identity_failed")
            d = LogDecision(d.kind, d.player_id, d.seat, d.generation, d.move_number, d.snapshot, d.payload,
                            "income_identity_failed")
        out.append(d)
    result.decisions = out
