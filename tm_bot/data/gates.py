"""Eligibility, setup, draft, and outcome validation gates for warm-start extraction."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Collection

from tm_bot.data.cards import CardRegistry
from tm_bot.data.inventory import (
    _extract_conceded,
    _extract_other_modules,
    _opt_bool,
    _opt_str,
    normalize_map,
)

SUPPORTED_MAPS: tuple[str, ...] = ("Tharsis", "Elysium", "Hellas", "Vastitas Borealis")
_SUPPORTED_MAP_SET: frozenset[str] = frozenset(SUPPORTED_MAPS)

# Scraper versions observed to lack structured starting_hand/cards_kept/draft fields, plus
# `v2025.08.15.1706`, which strips the `M€` payment text the v2 income and research-buy gates need.
DEFAULT_EXCLUDED_SCRAPER_VERSIONS: frozenset[str] = frozenset(
    {
        "v2025.08.15.1706",
        "2025july30",
        "v2025.07.29.0100",
        "v2025.07.29.1618",
        "v2025.07.30.1221",
        "v2025.07.30.1624",
        "v2025.07.30.1933",
        "v2025.08.02.1018",
        "v2025.08.02.1811",
        "v2025.08.03.0023",
        "v2025.08.06.0942-dev",
        "v2025.08.06.1719",
        "v2025.08.06.Glados",
    }
)

_FILENAME_TABLE_RE = re.compile(r"^game_([^_]+)(?:_([^.]+))?\.json$", re.IGNORECASE)
_MODULE_RE = re.compile(r"Module:\s*([^|\r\n]+)", re.IGNORECASE)
_ALLOWED_SETUP_MODULES = frozenset({"corporate era", "prelude"})


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reason: str | None = None
    detail: str | None = None


def _extract_scraper_version(data: dict[str, Any]) -> str | None:
    meta = data.get("metadata")
    meta_dict = meta if isinstance(meta, dict) else {}
    return (
        _opt_str(meta_dict.get("scraper_version"))
        or _opt_str(meta_dict.get("parser_version"))
        or _opt_str(data.get("scraper_version"))
        or _opt_str(data.get("parser_version"))
    )


def _check_table_id_consistent(table_id: str | None, path: str) -> bool:
    if not table_id:
        return False
    fname = path.replace("\\", "/").rsplit("/", 1)[-1]
    match = _FILENAME_TABLE_RE.match(fname)
    if match is None:
        return False
    return match.group(1) == table_id


def _has_forbidden_setup_module(data: dict[str, Any], moves: Any) -> str | None:
    other = _extract_other_modules(data, moves)
    if other:
        return ", ".join(other)
    if isinstance(moves, list):
        for move in moves[:5]:
            if not isinstance(move, dict):
                continue
            desc = move.get("description")
            if not isinstance(desc, str):
                continue
            for match in _MODULE_RE.findall(desc):
                mod_name = re.sub(r"\s+", " ", match).strip().lower()
                if mod_name and mod_name not in _ALLOWED_SETUP_MODULES:
                    return mod_name
    return None


def check_index_row(
    row: dict[str, Any],
    *,
    excluded_versions: Collection[str | None] = DEFAULT_EXCLUDED_SCRAPER_VERSIONS,
    version_predicate: Callable[[str | None], bool] | None = None,
) -> GateResult:
    """Apply fast metadata-only eligibility gates to one SQLite index row."""
    if row.get("status") != "ok":
        return GateResult(False, "corrupt_or_unreadable", row.get("error_type"))

    if row.get("player_count") != 2:
        return GateResult(False, "not_two_players", str(row.get("player_count")))

    norm_map = normalize_map(row.get("map") or row.get("map_raw"))
    if norm_map not in _SUPPORTED_MAP_SET:
        return GateResult(False, "unsupported_map", str(norm_map))

    if row.get("draft_on") != 1:
        return GateResult(False, "draft_not_on", str(row.get("draft_on")))
    if row.get("prelude_on") != 1:
        return GateResult(False, "prelude_not_on", str(row.get("prelude_on")))
    if row.get("corporate_era_on") != 1:
        return GateResult(False, "corporate_era_not_on", str(row.get("corporate_era_on")))
    if row.get("colonies_on") != 0:
        return GateResult(False, "colonies_not_off", str(row.get("colonies_on")))

    if _opt_bool(row.get("beginners_corporations_on")) is True:
        return GateResult(False, "other_module_present", "beginners_corporations")
    raw_other = row.get("other_modules")
    if raw_other not in (None, "", "[]"):
        try:
            parsed_other = json.loads(raw_other) if isinstance(raw_other, str) else list(raw_other)
        except (ValueError, TypeError):
            parsed_other = [str(raw_other)]
        if parsed_other:
            return GateResult(False, "other_module_present", ", ".join(map(str, parsed_other)))

    table_id = _opt_str(row.get("table_id"))
    path = str(row.get("path") or "")
    if not _check_table_id_consistent(table_id, path):
        return GateResult(False, "missing_or_inconsistent_table_id", f"table_id={table_id!r}, path={path!r}")

    ver = _opt_str(row.get("scraper_version"))
    if ver in excluded_versions or (version_predicate is not None and not version_predicate(ver)):
        return GateResult(False, "excluded_scraper_version", str(ver))

    return GateResult(True)


def check_eligibility(
    data: dict[str, Any],
    *,
    path: str,
    card_registry: CardRegistry,
    excluded_versions: Collection[str | None] = DEFAULT_EXCLUDED_SCRAPER_VERSIONS,
    version_predicate: Callable[[str | None], bool] | None = None,
) -> GateResult:
    """Validate format, identity, version, and card-name gates on one parsed game record."""
    if not isinstance(data, dict):
        return GateResult(False, "corrupt_or_unreadable", type(data).__name__)

    players = data.get("players")
    if not isinstance(players, dict) or len(players) != 2:
        return GateResult(False, "not_two_players")

    norm_map = normalize_map(_opt_str(data.get("map")))
    if norm_map not in _SUPPORTED_MAP_SET:
        return GateResult(False, "unsupported_map", str(norm_map))

    if _opt_bool(data.get("draft_on")) is not True:
        return GateResult(False, "draft_not_on")
    if _opt_bool(data.get("prelude_on")) is not True:
        return GateResult(False, "prelude_not_on")
    if _opt_bool(data.get("corporate_era_on")) is not True:
        return GateResult(False, "corporate_era_not_on")
    if _opt_bool(data.get("colonies_on")) is not False:
        return GateResult(False, "colonies_not_off")

    moves = data.get("moves")
    forbidden_mod = _has_forbidden_setup_module(data, moves)
    if forbidden_mod is not None:
        return GateResult(False, "other_module_present", forbidden_mod)

    table_id = _opt_str(data.get("table_id"))
    if not _check_table_id_consistent(table_id, path):
        return GateResult(False, "missing_or_inconsistent_table_id", f"table_id={table_id!r}, path={path!r}")

    ver = _extract_scraper_version(data)
    if ver in excluded_versions or (version_predicate is not None and not version_predicate(ver)):
        return GateResult(False, "excluded_scraper_version", str(ver))

    # Card-name resolvability across player starting_hand, corporation, and setup/draft fields
    for pid, pinfo in players.items():
        if not isinstance(pinfo, dict):
            return GateResult(False, "corrupt_or_unreadable", f"player {pid} is not dict")
        corp = pinfo.get("corporation")
        if corp is not None and card_registry.resolve(corp) is None:
            return GateResult(False, "unresolvable_card_name", str(corp))
        sh = pinfo.get("starting_hand")
        if isinstance(sh, dict):
            for key in ("corporations", "preludes", "project_cards"):
                cards = sh.get(key)
                if isinstance(cards, list):
                    for raw_name in cards:
                        if card_registry.resolve(raw_name) is None:
                            return GateResult(False, "unresolvable_card_name", str(raw_name))

    if isinstance(moves, list):
        for move in moves:
            if not isinstance(move, dict):
                continue
            kept = move.get("cards_kept")
            if isinstance(kept, dict):
                for cards in kept.values():
                    if isinstance(cards, list):
                        for raw_name in cards:
                            if card_registry.resolve(raw_name) is None:
                                return GateResult(False, "unresolvable_card_name", str(raw_name))
            opts = move.get("card_options")
            if isinstance(opts, dict):
                for cards in opts.values():
                    if isinstance(cards, list):
                        for raw_name in cards:
                            if card_registry.resolve(raw_name) is None:
                                return GateResult(False, "unresolvable_card_name", str(raw_name))
            drafted = move.get("card_drafted")
            if drafted is not None and card_registry.resolve(drafted) is None:
                return GateResult(False, "unresolvable_card_name", str(drafted))

    return GateResult(True)


@dataclass(frozen=True)
class PlayerSetup:
    player_id: str
    player_name: str | None
    seat: int
    offered_corporations: tuple[str, ...]
    offered_preludes: tuple[str, ...]
    offered_projects: tuple[str, ...]
    chosen_corporation: str
    chosen_preludes: tuple[str, ...]
    chosen_projects: tuple[str, ...]


@dataclass(frozen=True)
class SetupGateResult:
    passed: bool
    reason: str | None = None
    detail: str | None = None
    players: tuple[PlayerSetup, ...] = ()


def _resolve_player_names(data: dict[str, Any]) -> dict[str, str | None]:
    players = data.get("players")
    if not isinstance(players, dict):
        return {}
    names: dict[str, str | None] = {}
    for pid, pinfo in players.items():
        if isinstance(pinfo, dict):
            names[str(pid)] = _opt_str(pinfo.get("name"))
        else:
            names[str(pid)] = None
    moves = data.get("moves")
    if isinstance(moves, list):
        for move in moves:
            if not isinstance(move, dict):
                continue
            pid = _opt_str(move.get("player_id"))
            pname = _opt_str(move.get("player_name"))
            if pid and pname and pid in names and names[pid] is None:
                names[pid] = pname
            if all(v is not None for v in names.values()):
                break
    return names


def _resolve_seats(data: dict[str, Any], player_names: dict[str, str | None]) -> dict[str, int]:
    players = data.get("players")
    pids = [str(pid) for pid in players.keys()] if isinstance(players, dict) else []
    if len(pids) != 2:
        return {}
    moves = data.get("moves")
    start_pid: str | None = None
    if isinstance(moves, list):
        for move in moves[:8]:
            if not isinstance(move, dict):
                continue
            desc = move.get("description")
            if isinstance(desc, str) and "is starting player" in desc.lower():
                mpid = _opt_str(move.get("player_id"))
                if mpid in pids:
                    start_pid = mpid
                    break
                for pid, pname in player_names.items():
                    if pname and f"{pname} is starting player" in desc:
                        start_pid = pid
                        break
                if start_pid is not None:
                    break
    if start_pid is None:
        start_pid = pids[0]
    other_pid = pids[1] if pids[0] == start_pid else pids[0]
    return {start_pid: 0, other_pid: 1}


def _resolve_card_list(raw_cards: Any, card_registry: CardRegistry) -> tuple[str, ...] | None:
    if not isinstance(raw_cards, list):
        return None
    resolved: list[str] = []
    for raw in raw_cards:
        canon = card_registry.resolve(raw)
        if canon is None:
            return None
        resolved.append(canon)
    return tuple(resolved)


_GEN_DRAFT_RE = re.compile(r"\bdrafts\s+\d+\s+cards?\b", re.IGNORECASE)


def _extract_setup_kept_by_player(
    moves: Any,
    *,
    card_registry: CardRegistry,
    offered_corps_by_pid: dict[str, set[str]],
) -> dict[str, list[str]]:
    """Extract confirmed initial setup cards_kept per player before gameplay begins."""
    kept_by_player: dict[str, list[str]] = {}
    if not isinstance(moves, list):
        return kept_by_player

    def _has_corporation(pid_str: str, cards: list[Any]) -> bool:
        if card_registry.card_kinds:
            return any(card_registry.card_kinds.get(card_registry.resolve(c) or "") == "corporation" for c in cards)
        corps = offered_corps_by_pid.get(pid_str)
        if corps:
            return any((card_registry.resolve(c) or "") in corps for c in cards)
        return True

    for move in moves[:15]:
        if not isinstance(move, dict):
            continue
        act = _opt_str(move.get("action_type"))
        if act in ("place_tile", "standard_project", "pass", "fund_award", "claim_milestone"):
            break
        desc = move.get("description")
        desc_low = desc.lower() if isinstance(desc, str) else ""
        if (
            " plays card " in desc_low
            or " places " in desc_low
            or " takes standard project" in desc_low
            or "end of the generation" in desc_low
            or _GEN_DRAFT_RE.search(desc_low)
        ):
            break
        raw_kept = move.get("cards_kept")
        if isinstance(raw_kept, dict):
            mpid = _opt_str(move.get("player_id"))
            broke_into_prelude = False
            for pid, cards in raw_kept.items():
                pid_str = str(pid)
                # Setup keep always includes at least 1 corp + 2 preludes (>= 3 cards).
                if not isinstance(cards, list) or len(cards) < 3:
                    continue
                has_corp = _has_corporation(pid_str, cards)
                # A 3-card prelude draw (e.g. Biolabs / Research Network) has no corporation card
                # and must not overwrite an already-confirmed setup keep.
                if pid_str in kept_by_player and not has_corp:
                    broke_into_prelude = True
                    continue
                # Allow the acting player to overwrite their own keep if they undid setup,
                # while not letting an opponent's move overwrite with cumulative draws (e.g. Point Luna).
                if pid_str not in kept_by_player or (mpid == pid_str and has_corp):
                    kept_by_player[pid_str] = cards
            if broke_into_prelude and len(kept_by_player) >= 2:
                break
    return kept_by_player


def extract_verified_setup(
    data: dict[str, Any],
    *,
    card_registry: CardRegistry,
) -> SetupGateResult:
    """Validate starting_hand offers vs setup cards_kept for both players."""
    players = data.get("players")
    if not isinstance(players, dict) or len(players) != 2:
        return SetupGateResult(False, "not_two_players")

    player_names = _resolve_player_names(data)
    seats = _resolve_seats(data, player_names)
    offered_corps_by_pid: dict[str, set[str]] = {}
    for pid, pinfo in players.items():
        if isinstance(pinfo, dict) and isinstance(pinfo.get("starting_hand"), dict):
            raw_corps = pinfo["starting_hand"].get("corporations")
            if isinstance(raw_corps, list):
                offered_corps_by_pid[str(pid)] = {
                    c for c in (card_registry.resolve(x) for x in raw_corps) if c is not None
                }
    kept_raw_by_player = _extract_setup_kept_by_player(
        data.get("moves"),
        card_registry=card_registry,
        offered_corps_by_pid=offered_corps_by_pid,
    )

    verified_players: list[PlayerSetup] = []
    for pid, pinfo in players.items():
        pid_str = str(pid)
        if not isinstance(pinfo, dict):
            return SetupGateResult(False, "invalid_starting_hand", pid_str)

        sh = pinfo.get("starting_hand")
        if not isinstance(sh, dict):
            return SetupGateResult(False, "invalid_starting_hand", f"missing starting_hand for {pid_str}")

        offered_corps = _resolve_card_list(sh.get("corporations"), card_registry)
        offered_preludes = _resolve_card_list(sh.get("preludes"), card_registry)
        offered_projects = _resolve_card_list(sh.get("project_cards"), card_registry)
        if (
            offered_corps is None
            or offered_preludes is None
            or offered_projects is None
            or len(offered_corps) != 2
            or len(set(offered_corps)) != 2
            or len(offered_preludes) != 4
            or len(set(offered_preludes)) != 4
            or len(offered_projects) != 10
            or len(set(offered_projects)) != 10
        ):
            return SetupGateResult(False, "invalid_starting_hand", f"bad starting_hand counts for {pid_str}")

        raw_kept = kept_raw_by_player.get(pid_str)
        if raw_kept is None:
            return SetupGateResult(False, "missing_setup_choices", f"missing setup cards_kept for {pid_str}")

        chosen_all = _resolve_card_list(raw_kept, card_registry)
        if chosen_all is None:
            return SetupGateResult(False, "unresolvable_card_name", f"in setup cards_kept for {pid_str}")

        if len(chosen_all) != len(set(chosen_all)):
            return SetupGateResult(False, "invalid_setup_counts", f"duplicate kept card for {pid_str}")

        corp_set = set(offered_corps)
        prelude_set = set(offered_preludes)
        project_set = set(offered_projects)
        offered_union = corp_set | prelude_set | project_set

        unoffered = [c for c in chosen_all if c not in offered_union]
        if unoffered:
            return SetupGateResult(
                False,
                "setup_chosen_not_subset_of_offered",
                f"{pid_str}: {unoffered}",
            )

        chosen_corps = tuple(c for c in chosen_all if c in corp_set)
        chosen_preludes = tuple(c for c in chosen_all if c in prelude_set)
        chosen_projects = tuple(c for c in chosen_all if c in project_set)

        if len(chosen_corps) != 1 or len(chosen_preludes) != 2 or not (0 <= len(chosen_projects) <= 10):
            return SetupGateResult(
                False,
                "invalid_setup_counts",
                f"{pid_str}: corps={len(chosen_corps)} preludes={len(chosen_preludes)} proj={len(chosen_projects)}",
            )

        raw_corp = pinfo.get("corporation")
        if raw_corp is not None:
            canon_corp = card_registry.resolve(raw_corp)
            if canon_corp != chosen_corps[0]:
                return SetupGateResult(
                    False,
                    "setup_corporation_mismatch",
                    f"{pid_str}: player.corporation={canon_corp!r} != kept={chosen_corps[0]!r}",
                )

        verified_players.append(
            PlayerSetup(
                player_id=pid_str,
                player_name=player_names.get(pid_str),
                seat=seats[pid_str],
                offered_corporations=offered_corps,
                offered_preludes=offered_preludes,
                offered_projects=offered_projects,
                chosen_corporation=chosen_corps[0],
                chosen_preludes=chosen_preludes,
                chosen_projects=chosen_projects,
            )
        )

    verified_players.sort(key=lambda p: p.seat)
    return SetupGateResult(True, players=tuple(verified_players))


@dataclass(frozen=True)
class VerifiedDraftPick:
    player_id: str
    seat: int
    move_number: int
    draft_round_id: int
    pick_index: int
    generation: int
    packet_cards: tuple[str, ...]
    chosen_card: str


@dataclass(frozen=True)
class DraftExtractionResult:
    picks: tuple[VerifiedDraftPick, ...] = ()
    dropped_count: int = 0
    dropped_reasons: tuple[str, ...] = ()


def _quarantine_inconsistent_rounds(
    picks: list[VerifiedDraftPick], dropped_reasons: list[str]
) -> list[VerifiedDraftPick]:
    """Keep only picks whose (player, draft round) sequence is consistent (#30).

    `draft_round_id` is the generation key (generation = round + 1), so each
    player's picks in a round must be distinct `pick_index` values forming a
    prefix 0, 1, 2, ...:
    - a pick outside any draft round (round 0) is dropped (`no_draft_round`);
    - a round with a repeated `pick_index` means two rounds were merged or a pick
      event was logged twice, so the whole round is dropped (`duplicate_pick_key`);
    - picks after a gap (an earlier pick failed verification) are dropped
      (`missing_earlier_pick`), since their drafted-so-far set is incomplete.
    """
    groups: dict[tuple[str, int], list[VerifiedDraftPick]] = {}
    for pick in picks:
        groups.setdefault((pick.player_id, pick.draft_round_id), []).append(pick)
    bad: set[int] = set()
    for (_, round_id), group in groups.items():
        indices = [p.pick_index for p in group]
        if round_id < 1:
            reason_by_pick = {id(p): "no_draft_round" for p in group}
        elif len(set(indices)) != len(indices):
            reason_by_pick = {id(p): "duplicate_pick_key" for p in group}
        else:
            reason_by_pick = {}
            expected = 0
            for p in sorted(group, key=lambda q: q.pick_index):
                if p.pick_index == expected:
                    expected += 1
                else:
                    reason_by_pick[id(p)] = "missing_earlier_pick"
                    expected = -1  # everything after a gap is dropped
        for pick_id, reason in reason_by_pick.items():
            bad.add(pick_id)
            dropped_reasons.append(reason)
    return [p for p in picks if id(p) not in bad]


def extract_verified_drafts(
    data: dict[str, Any],
    *,
    card_registry: CardRegistry,
) -> DraftExtractionResult:
    """Extract draft picks where chosen card is strictly in the verified pre-pick packet."""
    players = data.get("players")
    if not isinstance(players, dict) or len(players) != 2:
        return DraftExtractionResult()
    moves = data.get("moves")
    if not isinstance(moves, list):
        return DraftExtractionResult()

    player_names = _resolve_player_names(data)
    seats = _resolve_seats(data, player_names)

    verified_picks: list[VerifiedDraftPick] = []
    round_picks: list[VerifiedDraftPick] = []
    pending_packet: dict[str, tuple[str, ...]] = {}
    dropped_reasons: list[str] = []
    draft_round_id = 0

    def _flush_round() -> None:
        if round_picks:
            verified_picks.extend(round_picks)
            round_picks.clear()
        pending_packet.clear()

    for idx, move in enumerate(moves):
        if not isinstance(move, dict):
            continue

        desc = move.get("description")
        desc_str = desc if isinstance(desc, str) else ""
        desc_low = desc_str.lower()
        act = _opt_str(move.get("action_type"))
        reason = _opt_str(move.get("reason"))
        pid = _opt_str(move.get("player_id"))
        raw_kept = move.get("cards_kept")
        has_kept = isinstance(raw_kept, dict) and any(isinstance(v, list) and v for v in raw_kept.values())

        # Any keep/buy move ends an active draft round and is never a draft pick itself
        if has_kept:
            _flush_round()
            continue

        raw_opts = move.get("card_options")
        raw_drafted = _opt_str(move.get("card_drafted"))

        # Draw-and-keep effects are not generation draft rounds
        if act == "draw":
            _flush_round()
            continue

        is_draft_pick_move = (
            raw_drafted is not None
            and (act == "draft" or reason == "draft")
            and " keeps " not in desc_low
        )

        if is_draft_pick_move:
            chosen = card_registry.resolve(raw_drafted)
            pkt = pending_packet.get(pid or "")
            if chosen is None:
                dropped_reasons.append("unresolvable_card_name")
            elif pid is None or pid not in seats:
                dropped_reasons.append("unknown_player")
            elif pkt is None or len(pkt) < 2:
                dropped_reasons.append("missing_pre_pick_packet")
            elif chosen not in pkt:
                dropped_reasons.append("chosen_not_in_packet")
            else:
                mv_num = move.get("move_number")
                mv_int = mv_num if isinstance(mv_num, int) and not isinstance(mv_num, bool) else idx
                round_picks.append(
                    VerifiedDraftPick(
                        player_id=pid,
                        seat=seats[pid],
                        move_number=mv_int,
                        draft_round_id=draft_round_id,
                        pick_index=max(0, 4 - len(pkt)),
                        generation=draft_round_id + 1,
                        packet_cards=pkt,
                        chosen_card=chosen,
                    )
                )

            # Apply post-pick swapped packet options if carried on this move
            if isinstance(raw_opts, dict):
                for opt_pid, opt_cards in raw_opts.items():
                    resolved_opt = _resolve_card_list(opt_cards, card_registry)
                    if resolved_opt is not None:
                        pending_packet[str(opt_pid)] = resolved_opt
                    else:
                        pending_packet.pop(str(opt_pid), None)

        elif raw_drafted is None and isinstance(raw_opts, dict) and (
            "drafts " in desc_low or act in ("draft", "pass")
        ):
            if not pending_packet and not round_picks:
                draft_round_id += 1
            for opt_pid, opt_cards in raw_opts.items():
                opid = str(opt_pid)
                resolved_opt = _resolve_card_list(opt_cards, card_registry)
                if resolved_opt is None:
                    pending_packet.pop(opid, None)
                    dropped_reasons.append("unresolvable_card_name")
                    continue
                prev_pkt = pending_packet.get(opid)
                if prev_pkt is not None and len(resolved_opt) >= len(prev_pkt):
                    # Undo / re-deal: invalidate buffered picks for this player at <= re-dealt size
                    kept_round: list[VerifiedDraftPick] = []
                    for rp in round_picks:
                        if rp.player_id == opid and len(rp.packet_cards) <= len(resolved_opt):
                            dropped_reasons.append("undone_draft_pick")
                        else:
                            kept_round.append(rp)
                    round_picks = kept_round
                pending_packet[opid] = resolved_opt

    _flush_round()
    kept_picks = _quarantine_inconsistent_rounds(verified_picks, dropped_reasons)
    return DraftExtractionResult(
        picks=tuple(kept_picks),
        dropped_count=len(dropped_reasons),
        dropped_reasons=tuple(dropped_reasons),
    )


@dataclass(frozen=True)
class OutcomeGateResult:
    passed: bool
    reason: str | None = None
    detail: str | None = None
    winner_player_id: str | None = None
    scores_by_player: dict[str, int] | None = None
    generations: int | None = None


def check_outcome(data: dict[str, Any]) -> OutcomeGateResult:
    """Validate non-concession, final_vp vs terminal scoring text, and winner consistency."""
    players = data.get("players")
    if not isinstance(players, dict) or len(players) != 2:
        return OutcomeGateResult(False, "not_two_players")

    moves = data.get("moves")
    if not isinstance(moves, list) or not moves:
        return OutcomeGateResult(False, "missing_terminal_scoring_text")

    # Check concession from both root flag and terminal move descriptions
    conceded = _extract_conceded(data, moves)
    if conceded is True:
        return OutcomeGateResult(False, "conceded")
    for move in moves[-3:]:
        if isinstance(move, dict):
            desc = move.get("description")
            if isinstance(desc, str) and "concedes the game" in desc.lower():
                return OutcomeGateResult(False, "conceded")
    if conceded is None:
        return OutcomeGateResult(False, "unknown_concession_status")

    player_names = _resolve_player_names(data)
    pids = [str(pid) for pid in players.keys()]
    if any(not player_names.get(pid) for pid in pids):
        return OutcomeGateResult(False, "missing_player_name")
    if player_names[pids[0]] == player_names[pids[1]]:
        return OutcomeGateResult(False, "ambiguous_player_name")

    name_to_pid = {player_names[pid]: pid for pid in pids if player_names[pid]}

    # Find terminal scoring move
    terminal_desc: str | None = None
    for move in reversed(moves[-3:]):
        if not isinstance(move, dict):
            continue
        desc = move.get("description")
        if isinstance(desc, str) and "TOTAL VP" in desc and "The end of the game:" in desc:
            terminal_desc = desc
            break
    if terminal_desc is None:
        return OutcomeGateResult(False, "missing_terminal_scoring_text")

    segments = [seg.strip() for seg in terminal_desc.split("|")]
    term_vp: dict[str, int] = {}
    term_mc: dict[str, int] = {}
    term_winner_pid: str | None = None

    for pid in pids:
        pname = player_names[pid]
        assert pname is not None
        vp_pat = re.compile(rf"^{re.escape(pname)}\s+scores\s+(-?\d+)\s+TOTAL VP$")
        mc_pat = re.compile(rf"^{re.escape(pname)}\s+has\s+(-?\d+)\s+M€\s+left\b")
        win_pat = re.compile(rf"^The end of the game:\s+{re.escape(pname)}\s+wins!$")
        for seg in segments:
            m_vp = vp_pat.match(seg)
            if m_vp:
                term_vp[pid] = int(m_vp.group(1))
            m_mc = mc_pat.match(seg)
            if m_mc:
                term_mc[pid] = int(m_mc.group(1))
            if win_pat.match(seg):
                term_winner_pid = pid

    if len(term_vp) != 2:
        return OutcomeGateResult(False, "missing_terminal_scoring_text")

    root_vp: dict[str, int] = {}
    for pid in pids:
        pinfo = players[pid]
        vp = pinfo.get("final_vp") if isinstance(pinfo, dict) else None
        if not isinstance(vp, int) or isinstance(vp, bool):
            return OutcomeGateResult(False, "missing_final_vp", pid)
        root_vp[pid] = vp

    for pid in pids:
        if root_vp[pid] != term_vp[pid]:
            return OutcomeGateResult(
                False,
                "score_text_mismatch",
                f"{pid}: root={root_vp[pid]} != terminal={term_vp[pid]}",
            )

    raw_winner = _opt_str(data.get("winner"))
    if raw_winner is None:
        return OutcomeGateResult(False, "missing_winner")

    root_winner_pid = name_to_pid.get(raw_winner)
    if root_winner_pid is None and raw_winner in players:
        root_winner_pid = raw_winner

    if root_winner_pid is None or term_winner_pid is None or root_winner_pid != term_winner_pid:
        return OutcomeGateResult(
            False,
            "winner_mismatch",
            f"root={raw_winner!r}, terminal_winner={term_winner_pid!r}",
        )

    p0, p1 = pids[0], pids[1]
    if root_vp[p0] != root_vp[p1]:
        expected_winner = p0 if root_vp[p0] > root_vp[p1] else p1
        if root_winner_pid != expected_winner:
            return OutcomeGateResult(
                False,
                "winner_mismatch",
                f"winner={root_winner_pid} does not have higher VP ({root_vp})",
            )
    else:
        if len(term_mc) == 2 and term_mc[p0] != term_mc[p1]:
            expected_winner = p0 if term_mc[p0] > term_mc[p1] else p1
            if root_winner_pid != expected_winner:
                return OutcomeGateResult(
                    False,
                    "winner_mismatch",
                    f"tie winner={root_winner_pid} disagrees with M€ tiebreaker ({term_mc})",
                )
        else:
            return OutcomeGateResult(False, "ambiguous_tie_winner")

    raw_gens = data.get("generations")
    gens = raw_gens if isinstance(raw_gens, int) and not isinstance(raw_gens, bool) else None

    return OutcomeGateResult(
        True,
        winner_player_id=root_winner_pid,
        scores_by_player=root_vp,
        generations=gens,
    )


@dataclass(frozen=True)
class RecordEvaluation:
    table_id: str | None
    perspective_player: str | None
    scraper_version: str | None
    map_name: str | None
    eligibility: GateResult
    setup: SetupGateResult
    drafts: DraftExtractionResult
    outcome: OutcomeGateResult


def evaluate_record(
    data: dict[str, Any],
    *,
    path: str,
    card_registry: CardRegistry,
    excluded_versions: Collection[str | None] = DEFAULT_EXCLUDED_SCRAPER_VERSIONS,
    version_predicate: Callable[[str | None], bool] | None = None,
) -> RecordEvaluation:
    """Run eligibility, setup, draft, and outcome gates on one parsed game record."""
    table_id = _opt_str(data.get("table_id")) if isinstance(data, dict) else None
    perspective_player = _opt_str(data.get("player_perspective")) if isinstance(data, dict) else None
    scraper_version = _extract_scraper_version(data) if isinstance(data, dict) else None
    map_name = normalize_map(_opt_str(data.get("map"))) if isinstance(data, dict) else None

    elig = check_eligibility(
        data,
        path=path,
        card_registry=card_registry,
        excluded_versions=excluded_versions,
        version_predicate=version_predicate,
    )
    if not elig.passed:
        return RecordEvaluation(
            table_id=table_id,
            perspective_player=perspective_player,
            scraper_version=scraper_version,
            map_name=map_name,
            eligibility=elig,
            setup=SetupGateResult(False, elig.reason, elig.detail),
            drafts=DraftExtractionResult(),
            outcome=OutcomeGateResult(False, elig.reason, elig.detail),
        )

    setup = extract_verified_setup(data, card_registry=card_registry)
    if not setup.passed:
        return RecordEvaluation(
            table_id=table_id,
            perspective_player=perspective_player,
            scraper_version=scraper_version,
            map_name=map_name,
            eligibility=elig,
            setup=setup,
            drafts=DraftExtractionResult(),
            outcome=OutcomeGateResult(False, setup.reason, setup.detail),
        )

    drafts = extract_verified_drafts(data, card_registry=card_registry)
    outcome = check_outcome(data)
    return RecordEvaluation(
        table_id=table_id,
        perspective_player=perspective_player,
        scraper_version=scraper_version,
        map_name=map_name,
        eligibility=elig,
        setup=setup,
        drafts=drafts,
        outcome=outcome,
    )
