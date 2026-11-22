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

Action-root and attribution gate (Gate 5, #44). A gen 2+ row is a root decision of seat `player_id`
when its own segments carry a root verb (`plays card`, `plays standard project`, `activates`,
`claims milestone`, `funds ... award`, `skips second action`, a bare `pays 8 Heat` / `pays 8 Plant`
conversion) or `action_type` is `pass` (`passes`, or the bundled pass + generation transition +
deal row, whose pass is observed on pre-transition state). The description pronoun "You" is never
used. Stripped and counted (`gate5:<reason>`), never emitted: `NAME other` rows, cancellations,
resignations (`concedes the game`; never a `Pass`), `skips rest of actions`, a root verb whose
segment names the other seat, unsupported standard projects and unclassifiable root rows. Each root
emits a `branch` decision and, for its kind, a `project_card`, `card_action` or `standard_project`
decision, with a menu from `tm_bot.data.actionmenu` built on the same pre-move state. A chosen
candidate outside its menu drops the row as `menu_miss:<reason>`; a menu with fewer than 2
candidates drops it as `single_candidate`. `branch`, `project_card` and `card_action` need the
exact tracked hand (forward quarantine `hand_lost`); `standard_project` only needs M€ (its menu
does not read the hand), so it is kept after a hand loss. Nested prompts that never emit rows are
counted per row as `excluded:<type>` (`payment` on paid roots, `player` targets, effect
`other_card` targets, gen-1 `prelude_play`).

Board replay and space rows (#61). On the supported maps (`tm_bot.data.board.BOARD_MAPS`) the
table's board is rebuilt in move order from every `places <tile> on|into <space>` segment (the
segment's actor owns the tile), or from the structured `tile_placed` / `tile_location` fields when a
row has no placement text. An unattributed, unparseable or unlocatable placement, or one on an
occupied space, forward-quarantines the board (`board_quarantine:<reason>`, counted for both
seats). After every row the board is checked against the row's snapshot: the ocean count must match
exactly (`check:oceans`); each seat's city and greenery counts from the VP details lag placements by
up to 10+ rows, so they only must never exceed the board's (`check:cities`, `check:greeneries`).
From the quarantine on, rows carry no board and emit no `space` rows (placements counted
`space:board_quarantined`). Each placement in gen 2+ emits one `space` decision on the board as of
the rows before it, with a menu from `tm_bot.data.spacemenu` for its source's rule (the root's card,
card action or standard project, a plant conversion including a bare `pays N Plant` row, the 0 C
temperature ocean, or a Hellas ocean-bonus space); `placement_index` numbers the placements of one
source. Drops: `unknown_source`, `land_claim_unknown` (Land Claim's marker space is never logged, so
the opponent's later placements are dropped), `no_prompt` (Noctis City on Tharsis, off-Mars
cities), `menu_miss:<rule>`, `single_candidate`. While the board is valid, gen 2+ action-phase
snapshots carry it as `board_tiles` / `board_owners` (`Board.compact`: engine tile code and owner
0 actor / 1 opponent / -1 per sorted space id). Each seat ends with one `board:valid_at_end`,
`board:quarantined` or `board:unsupported_map` count.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Collection, Mapping

from tm_bot.agents.card_data import CardSpec, load_card_data
from tm_bot.data import actionmenu, spacemenu
from tm_bot.data.board import (
    CITY_TILES,
    GREENERY,
    NO_TILE,
    OCEAN,
    OCEAN_BONUS,
    BOARD_MAPS,
    Board,
    Placement,
    geometry,
    locate,
    parse_placement,
    structured_placement,
)
from tm_bot.data.cards import CardRegistry
from tm_bot.data.gates import DraftExtractionResult, SetupGateResult, VerifiedDraftPick
from tm_bot.data.inventory import _opt_str, normalize_map

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
_STANDARD_PROJECT_RE = re.compile(r"^plays standard project (.+)$")
_FUNDS_AWARD_RE = re.compile(r"^funds (.+) award$")
_CONVERT_RE = re.compile(r"^pays (\d+) (Heat|Plant)$")
_PLACES_RE = re.compile(r"^places .+ (?:on|into) ")
_ANY_PLACES_RE = re.compile(r"(?:^| )places .+ (?:on|into) ")
_TARGET_VERBS = ("reduces ", "loses ", "removes ")
_RESIGN_MARKERS = ("concedes the game", "resigns")
ACTION_TASKS: tuple[str, ...] = ("branch", "project_card", "card_action", "standard_project", "space")
# Root -> the placement source its nested tile placements come from (`spacemenu.placement_sources`).
_SOURCE_ROOTS = frozenset({"play_card", "card_action", "standard_project"})


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
    kind: str  # "draft_pick" | "research_buy" | "deal_boundary" | "action" | one of ACTION_TASKS
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
    board_lost: tuple[int, str] | None = None  # (move, reason) of the board's forward quarantine

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
    # Action-phase public state of the current generation (#44).
    used_actions: set[str] = field(default_factory=set)
    turn_actions: int = 0
    passed: bool = False
    last_played: str | None = None
    # Tile placements (#61): the current root's placement source, its pre-move action context and the
    # space of the root's last placement (a Hellas ocean-bonus space grants a further ocean).
    place_source: str | None = None
    place_ctx: dict[str, Any] | None = None
    last_place: str | None = None
    place_count: int = 0  # placements the current root's source has made so far
    temperature_ocean: bool = False  # raised the temperature to 0 C: the engine grants an ocean
    land_claimed: bool = False  # played Land Claim: its (unlogged) marker closes one space to the opponent


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
    cards_by_name: Mapping[str, CardSpec] | None = None,
) -> LogStateResult:
    """Single pass over one perspective's moves; see the module docstring for the rules."""
    result = LogStateResult()
    specs = cards_by_name if cards_by_name is not None else load_card_data()
    map_name = (normalize_map(_opt_str(data.get("map"))) or "").lower()
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

    board = Board(geometry(BOARD_MAPS[map_name])) if map_name in BOARD_MAPS else None
    sources = spacemenu.placement_sources(specs)

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
        # The rebuilt board rides on gen 2+ action-phase rows while it is valid (#61).
        tiles, owners = (board.compact(actor.seat) if phase == "action" and gen >= 2 and board_valid()
                         else ([], []))
        snap["board_tiles"], snap["board_owners"] = tiles, owners
        return snap

    def board_valid() -> bool:
        return board is not None and result.board_lost is None

    def lose_board(move: int, reason: str) -> None:
        """Forward quarantine: the template board from `move` on, no `space` rows (counted per seat)."""
        if board_valid():
            result.board_lost = (move, reason)
            for st in seats.values():
                result.count(st.pid, f"board_quarantine:{reason}")

    def emit(kind: str, st: _Seat, move: int, gs: dict[str, Any], phase: str, payload: dict[str, Any],
             drop: str | None = None, snap: dict[str, Any] | None = None) -> None:
        result.decisions.append(
            LogDecision(kind=kind, player_id=st.pid, seat=st.seat, generation=gen, move_number=move,
                        snapshot=snap if snap is not None else snapshot(st, gs, phase), payload=payload,
                        drop_reason=drop)
        )

    def classify_root(st: _Seat, act: str | None, segs: list[str], deal: bool,
                      nxt: dict[str, Any] | None) -> tuple[str, Any] | None:
        """Gate 5: `(root, detail)`, `("strip", reason)`, or None for a non-root row of seat `st`.

        A bare `pays N Plant` row is also how an attacked seat loses plants, so it is a greenery
        conversion only when the seat's next row places the greenery (`nxt`, a label-side lookahead
        that never feeds a feature); a bare `pays 8 Heat` needs the temperature raise on its row.
        """
        parts = [split_actor(seg) for seg in segs]
        rests = [rest for _, rest in parts]
        if any(marker in rest for rest in rests for marker in _RESIGN_MARKERS):
            return "strip", "resignation"
        if any(rest == "cancels their move" for rest in rests):
            return "strip", "cancel"
        if any(rest.startswith("skips rest of actions") for rest in rests):
            return "strip", "skip_rest"
        if len(parts) == 1 and rests[0] == "other":
            return "strip", "other_row"
        for who, rest in parts:
            root: tuple[str, Any] | None = None
            if rest.startswith("plays card ") and act in ("play_card", "pass"):
                card = card_registry.resolve(rest[len("plays card "):].strip())
                root = ("play_card", card) if card is not None else ("strip", "unresolvable_card")
            elif (mt := _STANDARD_PROJECT_RE.match(rest)) and act == "standard_project":
                name = mt.group(1).strip().lower()
                if name == "sell patents":
                    root = ("sell_patents", None)
                elif name in actionmenu.LOG_STANDARD_PROJECTS:
                    root = ("standard_project", actionmenu.LOG_STANDARD_PROJECTS[name])
                else:
                    root = ("strip", "unsupported_standard_project")
            elif rest.startswith("activates ") and act == "activate_card":
                card = card_registry.resolve(rest[len("activates "):].strip())
                root = ("card_action", card) if card is not None else ("strip", "unresolvable_card")
            elif rest.startswith("claims milestone ") and act == "claim_milestone":
                root = ("claim_milestone", rest[len("claims milestone "):].strip())
            elif (mt := _FUNDS_AWARD_RE.match(rest)) and act == "fund_award":
                root = ("fund_award", mt.group(1).strip())
            elif rest.startswith("skips second action"):
                root = ("end_turn", None)
            if root is not None:
                return root if who is st or root[0] == "strip" else ("strip", "attribution_mismatch")
        who, rest = parts[0] if parts else (None, "")
        conv = _CONVERT_RE.match(rest) if who is st and act in ("other", None) else None
        if act == "convert_heat" or (conv and conv.group(2) == "Heat"
                                     and any(w is st and r.startswith("increases Temperature") for w, r in parts)):
            return "convert_heat", None
        nxt_segs = _segments(nxt.get("description")) if nxt and _opt_str(nxt.get("player_id")) == st.pid else []
        nxt_who, nxt_rest = split_actor(nxt_segs[0]) if nxt_segs else (None, "")
        places_greenery = nxt_who is st and nxt_rest.startswith("places Greenery")
        if act == "convert_plants" or (conv and conv.group(2) == "Plant" and places_greenery):
            return "convert_plants", int(conv.group(1)) if conv else None
        if act == "pass" and (deal or any(w is st and r == "passes" for w, r in parts)):
            return "pass", None
        if act in ROOT_ACTION_TYPES or act == "pass":
            return "strip", "unclassified_root"
        return None

    def action_decisions(st: _Seat, move: int, gs_prev: dict[str, Any], root: str, detail: Any) -> None:
        """Emit the `branch` row and the root's sub-task row from pre-move state (rows <= move - 1)."""
        opp = next(s for s in seats.values() if s.pid != st.pid)
        snap = snapshot(st, gs_prev, "action")
        exact = st.hand_lost_at is None
        # Public action-phase state the live observation carries (`actionsTakenThisRound`, `passed`,
        # `usedActions`); identical at the root and at its nested prompt.
        ctx = {"turn_action": st.turn_actions, "opp_passed": opp.passed,
               "me_used_actions": sorted(st.used_actions), "opp_used_actions": sorted(opp.used_actions)}
        st.place_ctx = ctx

        def finish(kind: str, payload: dict[str, Any], drop: str | None) -> None:
            if drop is not None:
                result.count(st.pid, f"{kind}:{drop}")
            emit(kind, st, move, gs_prev, "action", {**payload, **ctx}, drop, snap=snap)

        chosen = actionmenu.ROOT_TO_BRANCH[root]
        menu: list[str] = []
        drop = None
        if not exact:
            drop = "hand_lost"
        else:
            menu = actionmenu.branch_menu(snap, map_name=map_name, specs=specs, used_actions=st.used_actions,
                                          turn_action=st.turn_actions, opp_passed=opp.passed,
                                          last_played=st.last_played)
            if root == "convert_plants" and detail is not None and detail != actionmenu.plants_for_greenery(snap):
                drop = "state_sanity:convert_amount"
            elif chosen == "end_turn" and chosen not in menu:
                # The platform logs `skips second action` after the opponent passed too; the engine
                # then offers no End Turn (the skip ends the seat's generation).
                drop = "menu_miss:end_turn_" + ("opponent_passed" if opp.passed else "first_action")
            elif chosen not in menu:
                drop = f"menu_miss:{chosen}"
            elif len(menu) < 2:
                drop = "single_candidate"
        finish("branch", {"menu": menu, "chosen": chosen}, drop)

        if root == "play_card":
            menu, drop = [], None
            if not exact:
                drop = "hand_lost"
            else:
                menu, unchecked = actionmenu.playable_project_cards(snap, specs, st.last_played)
                spec = specs.get(detail)
                if detail not in st.hand or spec is None:
                    drop = "menu_miss:not_in_hand"
                elif (failure := actionmenu.card_playability(spec, snap, st.last_played)[0]) is not None:
                    drop = f"menu_miss:{failure}"
                elif len(menu) < 2:
                    drop = "single_candidate"
                if drop is None and unchecked:
                    result.count(st.pid, "info:project_card_menu_has_unchecked_requirement")
            finish("project_card", {"menu": menu, "chosen": detail}, drop)
        elif root == "card_action":
            menu, drop = [], None
            if not exact:
                drop = "hand_lost"
            else:
                menu = actionmenu.unused_action_cards(snap, specs, st.used_actions)
                spec = specs.get(detail)
                if detail not in st.tableau:
                    drop = "menu_miss:not_in_tableau"
                elif spec is None or not spec.has_action:
                    drop = "menu_miss:no_action"
                elif detail in st.used_actions:
                    drop = "menu_miss:used"
                elif len(menu) < 2:
                    drop = "single_candidate"
            finish("card_action", {"menu": menu, "chosen": detail}, drop)
        elif root == "standard_project":
            menu = actionmenu.standard_project_menu(snap)
            drop = "menu_miss:unaffordable" if detail not in menu else "single_candidate" if len(menu) < 2 else None
            finish("standard_project", {"menu": menu, "chosen": detail}, drop)

    def count_excluded(st: _Seat, act: str | None, segs: list[str], root: tuple[str, Any] | None) -> None:
        """Count the nested prompts a row shows evidence of; they never emit warm-start rows (#38 2D)."""
        parts = [split_actor(seg) for seg in segs]
        if root is not None and root[0] in ("play_card", "standard_project", "claim_milestone", "fund_award",
                                            "card_action") and any(
                who is st and rest.startswith("pays ") for who, rest in parts):
            result.count(st.pid, "excluded:payment")
        if any(who is not None and who is not st and rest.startswith(_TARGET_VERBS) for who, rest in parts):
            result.count(st.pid, "excluded:player")
        if root is None and act == "other" and any(who is st and rest.startswith("adds ") for who, rest in parts):
            result.count(st.pid, "excluded:other_card")

    def opp_has_passed(st: _Seat) -> bool:
        return any(s.passed for s in seats.values() if s is not st)

    def after_root(st: _Seat, root: str, detail: Any) -> None:
        """Advance the generation's turn state once a root decision is applied."""
        for other in seats.values():
            if other is not st:
                other.turn_actions = 0
        if root == "end_turn":
            st.turn_actions = 0
        elif root == "pass":
            st.passed, st.turn_actions, st.last_played = True, 0, None
        else:
            # Engine `takeAction`: the turn ends after 2 actions only while the opponent is still in the
            # generation; once it has passed, `actionsTakenThisRound` keeps counting up.
            st.turn_actions += 1
            if st.turn_actions >= 2 and not opp_has_passed(st):
                st.turn_actions = 0
        if root == "card_action":
            st.used_actions.add(detail)

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

    def set_place_source(st: _Seat, source: str | None) -> None:
        """A new root ends every seat's previous root; its nested placements come from `source`."""
        for other in seats.values():
            other.place_source, other.last_place, other.place_count = None, None, 0
        st.place_source = source
        if source is None:
            st.place_ctx = None

    def place_tile(who: _Seat | None, placement: Placement | None, tile_location: Any, move: int,
                   gs_prev: dict[str, Any] | None) -> None:
        """Apply one placement, emitting its `space` row first (state = the board before it).

        An unparseable, unattributable or unlocatable placement, or one on an occupied space,
        forward-quarantines the board (`board_quarantine:<reason>`).
        """
        assert board is not None
        if who is None:
            return lose_board(move, "unattributed_placement")
        if placement is None or placement.kind is None:
            return lose_board(move, "unparseable_placement")
        sid = locate(board.geo, placement, tile_location)
        if sid is None:
            return lose_board(move, "unlocatable_placement")
        if board.tile(sid) != NO_TILE:
            return lose_board(move, "occupied_space")
        src = who.place_source
        entry = sources.get(src) if src is not None else None
        if (entry is None or entry[0] != placement.kind) and placement.kind == "ocean" and who.last_place is not None \
                and OCEAN_BONUS in (int(b) for b in board.geo.spaces[who.last_place]["bonus"]):
            src, entry = spacemenu.HELLAS_OCEAN_BONUS, sources[spacemenu.HELLAS_OCEAN_BONUS]
        elif (entry is None or entry[0] != placement.kind) and placement.kind == "ocean" and who.temperature_ocean:
            src, entry = spacemenu.TEMPERATURE_OCEAN_BONUS, sources[spacemenu.TEMPERATURE_OCEAN_BONUS]
            who.temperature_ocean = False
        if entry is not None and entry[0] != placement.kind:
            entry = None
        if gen >= 2 and gs_prev is not None:
            snap = snapshot(who, gs_prev, "action")
            menu: list[str] = []
            rule = entry[1] if entry is not None else None
            if entry is None:
                drop: str | None = "unknown_source"
            elif any(s.land_claimed for s in seats.values() if s is not who):
                drop = "land_claim_unknown"
            elif rule == spacemenu.NO_PROMPT or (rule == "noctis" and board.geo.noctis is not None):
                drop = "no_prompt"
            else:
                menu = spacemenu.space_menu(rule, board, who.seat, actionmenu.funds_mc(snap))
                drop = f"menu_miss:{rule}" if sid not in menu else "single_candidate" if len(menu) < 2 else None
            if drop is not None:
                result.count(who.pid, f"space:{drop}")
            opp = next(s for s in seats.values() if s is not who)
            ctx = who.place_ctx or {"turn_action": who.turn_actions, "opp_passed": opp.passed,
                                    "me_used_actions": sorted(who.used_actions),
                                    "opp_used_actions": sorted(opp.used_actions)}
            emit("space", who, move, gs_prev, "action",
                 {"menu": menu, "chosen": sid, "rule": rule, "source": src, "tile": placement.tile,
                  "placement_index": who.place_count if src == who.place_source else 0, **ctx},
                 drop, snap=snap)
        board.place(sid, spacemenu.tile_code(placement.kind, src), who.seat)
        who.last_place = sid
        who.place_count += 1 if src == who.place_source else 0

    def check_board(move: int, gs: dict[str, Any]) -> None:
        """Gate the board against the row's snapshot. The ocean count posts on the placement row and
        must match exactly. A seat's city and greenery VP details can lag the placement by many rows,
        so they must never exceed the board's counts (a missed or misattributed placement shows up as
        the snapshot running ahead once the details catch up)."""
        assert board is not None
        if board.count({OCEAN}) != _int(gs.get("oceans")):
            return lose_board(move, "check:oceans")
        vp = gs.get("player_vp") or {}
        for st in seats.values():
            pvp = vp.get(st.pid) or {}
            if len((pvp.get("details") or {}).get("cities") or {}) > board.count(CITY_TILES, st.seat):
                return lose_board(move, "check:cities")
            if _int((pvp.get("total_details") or {}).get("greeneries")) > board.count({GREENERY}, st.seat):
                return lose_board(move, "check:greeneries")
        return None

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

        # Action-phase roots (Gate 5). A bundled deal row's pass is observed here, on the
        # pre-transition state of the generation it ends (`gen` is not advanced yet).
        if prev_gs is not None and gen >= 2 and pid in seats:
            st = seats[pid]
            root = classify_root(st, act, segs, is_deal, moves[idx + 1] if idx + 1 < len(moves) else None)
            if root is not None and root[0] == "strip":
                result.count(pid, f"gate5:{root[1]}")
                if root[1] == "skip_rest":
                    st.turn_actions = 0
                if root[1] != "other_row":  # effect-resolution rows sit between a play and its placement
                    set_place_source(st, None)
                root = None
            elif root is not None:
                if research_open and not is_deal:
                    end_research_window(move)
                if is_deal:
                    result.count(pid, "gate5:bundled_pass_split")
                action_decisions(st, move, prev_gs, *root)
                after_root(st, *root)
                set_place_source(st, root[1] if root[0] in _SOURCE_ROOTS
                                 else spacemenu.CONVERT_PLANTS if root[0] == "convert_plants" else None)
            elif (segs and (conv := _CONVERT_RE.match(split_actor(segs[0])[1])) and conv.group(2) == "Plant"
                  and split_actor(segs[0])[0] is st):
                # A bare `pays N Plant` row that Gate 5 does not take as a root (its greenery is logged as
                # `places Forest` / `places tile Forest`, or it is a final greenery) still sources the
                # seat's next greenery placement.
                set_place_source(st, spacemenu.CONVERT_PLANTS)
            count_excluded(st, act, segs, root)

        # Tile placements, in segment order on the board as of the rows before (#61).
        if prev_gs is not None and pid in seats and _int(prev_gs.get("temperature")) < 0 <= _int(gs.get("temperature")):
            seats[pid].temperature_ocean = True
        if board_valid():
            placements = [(who, rest) for who, rest in map(split_actor, segs)
                          if _PLACES_RE.match(rest) or (who is None and _ANY_PLACES_RE.search(rest))]
            for who, rest in placements:
                place_tile(who, parse_placement(rest) if who is not None else None,
                           m.get("tile_location") if len(placements) == 1 else None, move, prev_gs)
            if not placements and m.get("tile_placed") and isinstance(m.get("tile_location"), str):
                # A few rows log the placement only in the structured fields (the row's seat places it).
                place_tile(seats.get(pid or ""), structured_placement(m["tile_placed"], m["tile_location"]),
                           m["tile_location"], move, prev_gs)
        elif board is not None and gen >= 2:
            for who, rest in map(split_actor, segs):
                if who is not None and _PLACES_RE.match(rest):
                    result.count(who.pid, "space:board_quarantined")

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
                seats[p].used_actions = set()
                seats[p].turn_actions, seats[p].passed, seats[p].last_played = 0, False, None
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
                who.last_played = card
                who.land_claimed = who.land_claimed or card == "Land Claim"
                if card in who.hand:
                    who.hand.discard(card)
                elif card in who.preludes_in_hand:
                    who.preludes_in_hand.discard(card)
                    result.count(who.pid, "excluded:prelude_play")
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
            for st in seats.values():
                st.place_source, st.place_ctx, st.last_place, st.place_count = None, None, None, 0
        if board_valid():
            check_board(move, gs)
        prev_gs = gs

    for p, chk in checks.items():
        result.income_checks[(p, chk.generation)] = chk.status or "unverified"
    for st in seats.values():
        result.count(st.pid, "board:valid_at_end" if board_valid() else "board:unsupported_map" if board is None
                     else "board:quarantined")
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
