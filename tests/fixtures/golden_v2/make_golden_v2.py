"""Generate the synthetic golden v2 table pair (`python tests/fixtures/golden_v2/make_golden_v2.py`).

The two files are hand-scripted below in the real `v2025.10.07.1844` row format: invented
players and ids, real card names, and the snapshot timing rules measured in the archive audit
(the generation and the deal land on the bundled pass row; stocks stay stale by the
production income until a seat's next action-phase change posts; research-buy payments do not
post). Both perspectives carry the same rows, like real pairs: they differ only in the "You"
pronoun on draft rows, the timestamps, and `metadata.parsed_at`.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TABLE_ID = "900000001"
A, B = "10000001", "10000002"
NAMES = {A: "Alice Testa", B: "Bram Fixture"}
MC = "M€"

TRACKER_KEYS = (
    "Count of Wild tags", "Hand Counter", "Count of Animal tags", "Count of Science tags",
    "Count of City tags", "Player Area Counter", "Heat", "Titanium", "Steel Production",
    "Heat Production", "Count of Power tags", "Count of Jovian tags", MC, "Steel",
    "Count of played Events cards", "Titanium Production", "Plant", "Energy",
    "Count of Plant tags", "Count of Space tags", "Count of Earth tags", "Microbe",
    "Count of Microbe tags", "Count of Building tags", "Plant Production",
    f"{MC} Production", "Energy Production",
)
STOCKS = (MC, "Steel", "Titanium", "Plant", "Energy", "Heat")

A_OFFER = {
    "corporations": ["CrediCor", "Ecoline"],
    "preludes": ["Allied Bank", "Biolab", "Donation", "Loan"],
    "project_cards": ["Mine", "Research", "Wave Power", "Comet", "Power Plant", "Search For Life",
                      "Development Center", "Inventors' Guild", "Mars University", "Algae"],
}
B_OFFER = {
    "corporations": ["Helion", "Teractor"],
    "preludes": ["Business Empire", "Dome Farming", "Galilean Mining", "Power Generation"],
    "project_cards": ["Heather", "Lichen", "Sponsors", "Earth Office", "Fuel Factory", "Fish",
                      "Food Factory", "Grass", "Imported Nitrogen", "Satellites"],
}
A_KEEP = ["CrediCor", "Allied Bank", "Donation", "Mine", "Research", "Wave Power", "Comet"]
B_KEEP = ["Teractor", "Business Empire", "Dome Farming", "Sponsors", "Earth Office", "Fuel Factory",
          "Lichen", "Heather"]


class Game:
    def __init__(self) -> None:
        zero = {k: 0 for k in TRACKER_KEYS}
        self.true = {A: dict(zero), B: dict(zero)}
        self.shown = {A: dict(zero), B: dict(zero)}
        self.tr = {A: 20, B: 20}
        self.gen = 1
        self.temp, self.oxy, self.oceans = -30, 0, 0
        self.milestones: dict = {}
        self.awards: dict = {}
        self.greeneries: dict = {A: {}, B: {}}
        self.vp_ms = {A: 0, B: 0}
        self.rows: list[dict] = []

    def snapshot(self, move_number: int) -> dict:
        vp = {}
        for pid in (A, B):
            det = {
                "tr": self.tr[pid], "awards": 0, "milestones": self.vp_ms[pid],
                "cities": 0, "greeneries": len(self.greeneries[pid]), "cards": 0,
            }
            entry: dict = {"total": sum(det.values()), "total_details": det}
            if self.greeneries[pid]:
                entry["details"] = {"greeneries": copy.deepcopy(self.greeneries[pid])}
            vp[pid] = entry
        return {
            "move_number": move_number,
            "generation": self.gen,
            "temperature": self.temp,
            "oxygen": self.oxy,
            "oceans": self.oceans,
            "player_vp": vp,
            "milestones": copy.deepcopy(self.milestones),
            "awards": copy.deepcopy(self.awards),
            "player_trackers": {pid: dict(self.shown[pid]) for pid in (A, B)},
        }

    def row(self, pid: str, action_type: str, description: str, *, effects=None, post=True,
            **fields) -> None:
        """Append one row; `effects` = {pid: {tracker: delta}}; `post` syncs changed stocks."""
        for epid, deltas in (effects or {}).items():
            for key, delta in deltas.items():
                if key == "TR":
                    self.tr[epid] += delta
                    continue
                self.true[epid][key] += delta
                if post or key not in STOCKS:
                    self.shown[epid][key] = self.true[epid][key]
        move_number = len(self.rows) + 1
        base = {
            "move_number": move_number, "timestamp": None, "player_id": pid,
            "player_name": NAMES[pid], "action_type": action_type, "description": description,
            "card_played": None, "card_drafted": None, "tile_placed": None, "tile_location": None,
            "card_options": None, "cards_kept": None, "reason": None,
        }
        base.update(fields)
        base["game_state"] = self.snapshot(move_number)
        self.rows.append(base)

    def production(self) -> None:
        """Generation transition: true stocks gain income; the shown trackers stay stale."""
        for pid in (A, B):
            t = self.true[pid]
            t[MC] += t[f"{MC} Production"] + self.tr[pid]
            t["Heat"] += t["Energy"] + t["Heat Production"]
            t["Energy"] = t["Energy Production"]
            t["Steel"] += t["Steel Production"]
            t["Titanium"] += t["Titanium Production"]
            t["Plant"] += t["Plant Production"]
        self.gen += 1


def _draft_round(g: Game, picks) -> None:
    for pid, card, options in picks:
        g.row(pid, "draft", f"{NAMES[pid]} drafts {card}", card_drafted=card, card_options=options, reason="draft")


def build() -> Game:
    g = Game()
    a, b = NAMES[A], NAMES[B]
    # ---- Generation 1: setup (moves 1-7) --------------------------------------------------
    g.row(A, "other", f"Module: Corporate Era | Map: Tharsis | {a} is starting player for this generation",
          card_options={A: A_OFFER["project_cards"], B: B_OFFER["project_cards"]})
    g.row(B, "other", f"{b} keeps 8 cards: " + ", ".join(B_KEEP), card_drafted="Teractor",
          cards_kept={B: B_KEEP},
          effects={B: {MC: 45, "Hand Counter": 7, "Player Area Counter": 1, "Count of Earth tags": 1}})
    g.row(A, "other", f"{a} keeps 7 cards: " + ", ".join(A_KEEP) + f" | {b} keeps 8 cards: "
          + ", ".join(B_KEEP) + f" | {a} pays 12 {MC}", card_drafted="CrediCor",
          cards_kept={A: A_KEEP, B: B_KEEP},
          effects={A: {MC: 45, "Hand Counter": 6, "Player Area Counter": 1}})
    g.row(A, "play_card", f"{a} plays card Allied Bank | {a} increases {MC} Production by 4 (immediate effect of Allied Bank) | {a} gains 3 {MC} (immediate effect of Allied Bank)",
          card_played="Allied Bank",
          effects={A: {MC: 3, f"{MC} Production": 4, "Count of Earth tags": 1, "Hand Counter": -1, "Player Area Counter": 1}})
    g.row(A, "play_card", f"{a} plays card Donation | {a} gains 21 {MC} (immediate effect of Donation)",
          card_played="Donation", effects={A: {MC: 21, "Hand Counter": -1, "Player Area Counter": 1}})
    g.row(B, "play_card", f"{b} plays card Business Empire | {b} pays 6 {MC} (immediate effect of Business Empire) | {b} increases {MC} Production by 6 (immediate effect of Business Empire)",
          card_played="Business Empire",
          effects={B: {MC: -6, f"{MC} Production": 6, "Count of Earth tags": 1, "Hand Counter": -1, "Player Area Counter": 1}})
    g.row(B, "play_card", f"{b} plays card Dome Farming | {b} increases {MC} Production by 2 (immediate effect of Dome Farming) | {b} increases Plant Production by 1 (immediate effect of Dome Farming)",
          card_played="Dome Farming",
          effects={B: {f"{MC} Production": 2, "Plant Production": 1, "Count of Plant tags": 1,
                       "Count of Building tags": 1, "Hand Counter": -1, "Player Area Counter": 1}})
    # ---- Generation 1: actions (moves 8-14) -------------------------------------------------
    g.row(A, "play_card", f"{a} pays 4 {MC} | {a} plays card Mine | {a} increases Steel Production by 1 (immediate effect of Mine)",
          card_played="Mine",
          effects={A: {MC: -4, "Steel Production": 1, "Count of Building tags": 1, "Hand Counter": -1, "Player Area Counter": 1}})
    g.row(B, "play_card", f"{b} pays 3 {MC} | {b} plays card Sponsors | {b} increases {MC} Production by 2 (immediate effect of Sponsors)",
          card_played="Sponsors",
          effects={B: {MC: -3, f"{MC} Production": 2, "Count of Earth tags": 1, "Hand Counter": -1, "Player Area Counter": 1}})
    g.row(A, "play_card", f"{a} pays 11 {MC} | {a} plays card Research", card_played="Research",
          reason="immediate effect of Research",
          effects={A: {MC: -11, "Count of Science tags": 2, "Hand Counter": -1, "Player Area Counter": 1}})
    g.row(A, "draw", f"{a} draws 2 cards: Asteroid, Tardigrades",
          cards_kept={A: ["Asteroid", "Tardigrades"]}, effects={A: {"Hand Counter": 2}})
    g.row(B, "other", f"{b} other", reason="Earth Office")
    g.row(B, "pass", f"{b} passes")
    g.temp = -28
    g.row(A, "play_card", f"{a} pays 14 {MC} | {a} plays card Asteroid | {a} increases Temperature by 1 step/s to a value of -28 | {a} gains 1 TR | {a} gains 2 Titanium (immediate effect of Asteroid)",
          card_played="Asteroid",
          effects={A: {MC: -14, "Titanium": 2, "TR": 1, "Count of played Events cards": 1, "Hand Counter": -1}})
    # ---- Gen 1 -> 2 (move 15): bundled pass + transition + deal + one seat's income text ----
    g.production()
    deal2 = {A: ["Solar Wind Power", "Media Group", "Cartel", "Bushes"],
             B: ["Trees", "Capital", "Adapted Lichen", "Nitrogen-Rich Asteroid"]}
    g.row(A, "pass", f"{a} passes | {a} drafts 4 cards: " + ", ".join(deal2[A]) + f" | {b} drafts 4 cards: "
          + ", ".join(deal2[B]) + f" | {b} gains 30 {MC} | {b} gains 1 Plant", card_options=deal2)
    # ---- Generation 2 draft (moves 16-21) --------------------------------------------------
    _draft_round(g, [
        (A, "Media Group", None),
        (B, "Capital", {A: ["Trees", "Adapted Lichen", "Nitrogen-Rich Asteroid"], B: ["Solar Wind Power", "Cartel", "Bushes"]}),
        (A, "Trees", None),
        (B, "Cartel", {A: ["Solar Wind Power", "Bushes"], B: ["Adapted Lichen", "Nitrogen-Rich Asteroid"]}),
        (A, "Bushes", None),
        (B, "Nitrogen-Rich Asteroid", {A: ["Adapted Lichen"], B: ["Solar Wind Power"]}),
    ])
    # ---- Research buys (moves 22-23): the payments do not post to the stale M€ tracker -----
    g.row(A, "buy_card", f"{a} keeps 2 cards: Media Group, Trees | {a} pays 6 {MC}", card_drafted="Media Group",
          cards_kept={A: ["Media Group", "Trees"]}, effects={A: {MC: -6, "Hand Counter": 2}}, post=False)
    g.row(B, "buy_card", f"{b} keeps 3 cards: Capital, Cartel, Solar Wind Power | {a} keeps 2 cards: Media Group, Trees | {b} pays 9 {MC}",
          card_drafted="Capital", cards_kept={B: ["Capital", "Cartel", "Solar Wind Power"], A: ["Media Group", "Trees"]},
          effects={B: {MC: -9, "Hand Counter": 3}}, post=False)
    # ---- Generation 2: actions (moves 24-36) -----------------------------------------------
    g.row(B, "play_card", f"{b} pays 5 {MC} | {b} plays card Cartel | {b} increases {MC} Production by 4 (immediate effect of Cartel)",
          card_played="Cartel",
          effects={B: {MC: -5, f"{MC} Production": 4, "Count of Earth tags": 1, "Hand Counter": -1, "Player Area Counter": 1}})
    g.row(A, "play_card", f"{a} pays 6 {MC} | {a} plays card Media Group", card_played="Media Group",
          effects={A: {MC: -6, "Count of Earth tags": 1, "Hand Counter": -1, "Player Area Counter": 1}})
    g.row(B, "standard_project", f"{b} plays standard project Greenery | {b} pays 23 {MC}", reason="Greenery",
          effects={B: {MC: -23}})
    g.oxy = 1
    g.greeneries[B]["Tharsis Hex 5,6"] = {"vp": 1}
    g.row(B, "other", f"{b} places Greenery on Tharsis Hex 5,6 (5,6) | {b} increases Oxygen Level by 1 step/s to a value of 1 | {b} gains 1 TR",
          effects={B: {"TR": 1}})
    g.row(B, "draw", f"{b} draws Kelp Farming", cards_kept={B: ["Kelp Farming"]}, effects={B: {"Hand Counter": 1}})
    g.row(A, "other", f"{a} cancels their move")
    g.milestones["Builder"] = {"claimed_by": a, "player_id": A, "move_number": len(g.rows) + 1, "timestamp": None}
    g.vp_ms[A] = 5
    g.row(A, "claim_milestone", f"{a} claims milestone Builder | {a} pays 8 {MC}", effects={A: {MC: -8}})
    g.awards["Banker"] = {"funded_by": b, "player_id": B, "move_number": len(g.rows) + 1, "timestamp": None}
    g.row(B, "fund_award", f"{b} funds Banker award | {b} pays 8 {MC}", effects={B: {MC: -8}})
    g.row(A, "play_card", f"{a} pays 4 {MC} | {a} plays card Tardigrades", card_played="Tardigrades",
          effects={A: {MC: -4, "Count of Microbe tags": 1, "Hand Counter": -1, "Player Area Counter": 1}})
    g.row(B, "pass", f"{b} passes")
    g.row(A, "activate_card", f"{a} activates Tardigrades | {a} adds Microbe to Tardigrades", effects={A: {"Microbe": 1}})
    g.row(A, "play_card", f"{a} pays 8 {MC} | {a} plays card Wave Power | {a} increases Energy Production by 1 (immediate effect of Wave Power)",
          card_played="Wave Power",
          effects={A: {MC: -8, "Energy Production": 1, "Count of Power tags": 1, "Hand Counter": -1, "Player Area Counter": 1}})
    g.row(A, "other", f"{a} other")
    # ---- Gen 2 -> 3 (move 37) ---------------------------------------------------------------
    g.production()
    deal3 = {A: ["Windmills", "Nuclear Power", "Strip Mine", "Lava Flows"],
             B: ["Pets", "Heat Trappers", "Moss", "Zeppelins"]}
    g.row(A, "pass", f"{a} passes | {a} drafts 4 cards: " + ", ".join(deal3[A]) + f" | {b} drafts 4 cards: "
          + ", ".join(deal3[B]), card_options=deal3)
    # ---- Generation 3 draft (moves 38-43) --------------------------------------------------
    _draft_round(g, [
        (A, "Strip Mine", None),
        (B, "Pets", {A: ["Heat Trappers", "Moss", "Zeppelins"], B: ["Windmills", "Nuclear Power", "Lava Flows"]}),
        (A, "Heat Trappers", None),
        (B, "Lava Flows", {A: ["Windmills", "Nuclear Power"], B: ["Moss", "Zeppelins"]}),
        (A, "Windmills", None),
        (B, "Moss", {A: ["Zeppelins"], B: ["Nuclear Power"]}),
    ])
    # ---- Research (move 44): Alice keeps one card (single-card form); Bram keeps nothing, and
    # a seat that buys nothing has no keep row in real logs.
    g.row(A, "buy_card", f"{a} keeps Strip Mine | {a} pays 3 {MC}", card_drafted="Strip Mine",
          cards_kept={A: ["Strip Mine"]}, effects={A: {MC: -3, "Hand Counter": 1}}, post=False)
    # ---- Generation 3: actions (moves 45-53) -----------------------------------------------
    g.row(B, "play_card", f"{b} pays 11 {MC} | {b} plays card Solar Wind Power | {b} increases Energy Production by 1 (immediate effect of Solar Wind Power) | {b} gains 2 Titanium (immediate effect of Solar Wind Power)",
          card_played="Solar Wind Power",
          effects={B: {MC: -11, "Energy Production": 1, "Titanium": 2, "Count of Science tags": 1, "Count of Space tags": 1,
                       "Count of Power tags": 1, "Hand Counter": -1, "Player Area Counter": 1}})
    g.temp = -26
    g.row(A, "play_card", f"{a} pays 21 {MC} | {a} plays card Comet | {a} increases Temperature by 1 step/s to a value of -26 | {a} gains 1 TR | {a} gains 4 {MC} (triggered effect of CrediCor)",
          card_played="Comet", reason="immediate effect of Comet",
          effects={A: {MC: -17, "TR": 1, "Count of played Events cards": 1, "Hand Counter": -1}})
    g.oceans = 1
    g.row(A, "place_tile", f"{a} increases Oceans by 1 step/s to a value of 1 | {a} gains 1 TR | {a} places Ocean on Tharsis Hex 4,1 (4,1) | {a} gains 2 {MC}",
          tile_placed="Ocean", tile_location="Tharsis Hex 4,1 (4,1)", reason="immediate effect of Comet",
          effects={A: {MC: 2, "TR": 1}})
    g.row(B, "standard_project", f"{b} plays standard project Sell patents", reason="Sell patents")
    g.row(B, "other", f"{b} discards 1 card/s (Sell patents) | {b} gains 1 {MC}", effects={B: {MC: 1, "Hand Counter": -1}})
    g.row(A, "standard_project", f"{a} plays standard project Power Plant | {a} pays 11 {MC}", reason="Power Plant",
          effects={A: {MC: -11}})
    g.row(A, "other", f"{a} increases Energy Production by 1", effects={A: {"Energy Production": 1}})
    g.row(B, "pass", f"{b} passes")
    g.row(A, "activate_card", f"{a} activates Tardigrades | {a} adds Microbe to Tardigrades", effects={A: {"Microbe": 1}})
    g.row(A, "play_card", f"{a} pays 13 {MC} | {a} plays card Trees | {a} increases Plant Production by 3 (immediate effect of Trees) | {a} gains 1 Plant (immediate effect of Trees)",
          card_played="Trees",
          effects={A: {MC: -13, "Plant Production": 3, "Plant": 1, "Count of Plant tags": 1, "Hand Counter": -1, "Player Area Counter": 1}})
    # ---- Terminal row (move 55) -------------------------------------------------------------
    final_desc = (
        f"{a} passes | It is the end of the generation 3 and Mars is terraformed! | "
        f"{a} scores 23 point/s for Terraforming Rating | {b} scores 21 point/s for Terraforming Rating | "
        f"{a} scores 5 point/s for milestone | {b} scores 1 points for Greenery tiles | "
        f"{a} has 16 {MC} left (for tiebreaker purposes) | {a} scores 28 TOTAL VP | "
        f"{b} has 46 {MC} left (for tiebreaker purposes) | {b} scores 22 TOTAL VP | "
        f"The end of the game: {a} wins!"
    )
    g.row(A, "pass", final_desc)
    return g


def perspective(rows: list[dict], persp: str, hour: int) -> dict:
    out_rows = []
    for i, row in enumerate(rows):
        r = copy.deepcopy(row)
        r["timestamp"] = f"{hour:02d}:{10 + i // 60:02d}:{i % 60:02d}"
        if r["action_type"] == "draft" and r["player_id"] == persp:
            r["description"] = f"You draft {r['card_drafted']}"
        # A real-log quirk: an opponent pick can also read "You draft ..." (attribute by player_id).
        if r["action_type"] == "draft" and r["player_id"] != persp and r["card_drafted"] in ("Cartel", "Heat Trappers"):
            r["description"] = f"You draft {r['card_drafted']}"
        out_rows.append(r)
    for r in out_rows:
        for entry in list(r["game_state"]["milestones"].values()) + list(r["game_state"]["awards"].values()):
            entry["timestamp"] = out_rows[entry["move_number"] - 1]["timestamp"]
    return {
        "replay_id": TABLE_ID,
        "player_perspective": persp,
        "game_date": "2026-01-02",
        "game_duration": "00:31",
        "winner": NAMES[A],
        "generations": 3,
        "conceded": False,
        "map": "Tharsis",
        "prelude_on": True,
        "colonies_on": False,
        "corporate_era_on": True,
        "draft_on": True,
        "beginners_corporations_on": False,
        "game_speed": "Real-time • Fast paced",
        "players": {
            A: {"player_id": A, "player_name": NAMES[A], "corporation": "CrediCor", "final_vp": 28, "final_tr": 23,
                "starting_hand": A_OFFER},
            B: {"player_id": B, "player_name": NAMES[B], "corporation": "Teractor", "final_vp": 22, "final_tr": 21,
                "starting_hand": B_OFFER},
        },
        "metadata": {"parsed_at": f"2026-01-0{3 if persp == A else 4}T10:00:00", "total_moves": len(out_rows),
                     "scraper_version": "v2025.10.07.1844"},
        "table_id": TABLE_ID,
        "moves": out_rows,
    }


def main() -> None:
    g = build()
    for persp, hour in ((A, 14), (B, 9)):
        out = HERE / persp / f"game_{TABLE_ID}_{persp}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(perspective(g.rows, persp, hour), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for pid in (A, B):
        print(NAMES[pid], {k: g.true[pid][k] for k in STOCKS}, "TR", g.tr[pid])


if __name__ == "__main__":
    main()
