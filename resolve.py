"""
Roster cache: turn the model's shirt numbers into player and team names.

Matched to the hybrid prompt set:
  card_event / substitution -> number + `crest` (the only team signal on a banner)
  lineup                    -> number + team_name printed on the graphic
  main_scoreboard           -> team codes snapped to canonical names, period from clock

Uruguay and Spain share nearly every shirt number (#8 is Valverde AND Fabian Ruiz),
so a bare number is not enough on the banners. When the team cannot be determined the
result is marked `ambiguous` and carries BOTH candidates rather than guessing.
"""
from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^A-Z0-9 ]", "", s.upper()).strip()


def _score(a: str, b: str) -> float:
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    last = b.split()[-1] if b.split() else b
    return max(SequenceMatcher(None, a, b).ratio(), SequenceMatcher(None, a, last).ratio())


@lru_cache(maxsize=32)
def load_roster(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def team_side(roster: dict, text: str | None, floor: float = 0.6) -> str | None:
    if not text:
        return None
    best, best_s = None, 0.0
    for side in ("home", "away"):
        t = roster["teams"][side]
        s = max(_score(text, t["name"]), _score(text, t.get("short", "")))
        if s > best_s:
            best, best_s = side, s
    return best if best_s >= floor else None


def lookup(roster: dict, number, side: str | None = None) -> dict:
    if number is None:
        return {"player_name": None, "team_name": None, "shirt_number": None,
                "resolved": False, "reason": "no-number"}
    pool = [p for p in roster["players"]
            if p.get("number") == number and (side is None or p["team"] == side)]
    if len(pool) == 1:
        p = pool[0]
        return {"player_name": p["name"], "team_name": roster["teams"][p["team"]]["name"],
                "shirt_number": number, "resolved": True,
                "reason": "number+team" if side else "number"}
    if len(pool) > 1:
        return {"player_name": None, "team_name": None, "shirt_number": number,
                "resolved": False, "reason": "ambiguous",
                "candidates": [{"player_name": p["name"],
                                "team_name": roster["teams"][p["team"]]["name"]}
                               for p in pool]}
    return {"player_name": None, "team_name": None, "shirt_number": number,
            "resolved": False, "reason": "not-in-squad"}


# --------------------------------------------------------------------------- card
def resolve_card_event(vlm: dict, roster: dict | None) -> dict:
    if roster is None:
        return {"team_name": None, "player_name": None,
                "shirt_number": vlm.get("number"), "card_type": vlm.get("card_type"),
                "minute": None, "_cache": {"reason": "no-roster"}}
    side = team_side(roster, vlm.get("crest"))
    r = lookup(roster, vlm.get("number"), side)
    team = r["team_name"] or (roster["teams"][side]["name"] if side else None)
    meta = {"crest": vlm.get("crest"), "team_side": side,
            "resolved": r["resolved"], "reason": r["reason"]}
    if "candidates" in r:
        meta["candidates"] = r["candidates"]
    return {"team_name": team, "player_name": r["player_name"],
            "shirt_number": r["shirt_number"], "card_type": vlm.get("card_type"),
            "minute": None, "_cache": meta}


# ------------------------------------------------------------------- substitution
def resolve_substitution(vlm: dict, roster: dict | None) -> dict:
    arrow = (vlm.get("arrow") or "").lower()
    default = {"green": "in", "red": "out"}.get(arrow)
    ins, outs, players = [], [], []
    side = team_side(roster, vlm.get("crest")) if roster else None
    for row in vlm.get("players") or []:
        d = row.get("direction") or default
        r = (lookup(roster, row.get("number"), side) if roster else
             {"player_name": None, "team_name": None,
              "shirt_number": row.get("number"), "resolved": False,
              "reason": "no-roster"})
        players.append({**r, "direction": d})
        label = r["player_name"] or f"#{row.get('number')}"
        if d == "in":
            ins.append(label)
        elif d == "out":
            outs.append(label)
    team = next((p["team_name"] for p in players if p.get("team_name")), None)
    if not team and side and roster:
        team = roster["teams"][side]["name"]
    dirs = {p["direction"] for p in players}
    return {"team_name": team, "player_in": ins, "player_out": outs,
            "direction": dirs.pop() if len(dirs) == 1 else ("mixed" if dirs else None),
            "players": players,
            "_cache": {"arrow": arrow or None, "crest": vlm.get("crest"),
                       "team_side": side,
                       "unresolved": [p["shirt_number"] for p in players
                                      if not p["resolved"]]}}


# ------------------------------------------------------------------------- lineup
def resolve_lineup(vlm: dict, roster: dict | None) -> dict:
    """The graphic prints the team name, so the squad is known and numbers resolve
    unambiguously. `name_text` is only a fallback when a number is unreadable."""
    side = team_side(roster, vlm.get("team_name")) if roster else None
    wrong_match = roster is not None and side is None and vlm.get("team_name")
    pool = ([p for p in roster["players"] if p["team"] == side] if (roster and side)
            else (roster["players"] if roster else []))

    def rows(key):
        out, hows = [], []
        for p in vlm.get(key) or []:
            num, txt = p.get("number"), p.get("name_text")
            cand = [x for x in pool if num is not None and x.get("number") == num]
            if len(cand) == 1:
                name, how = cand[0]["name"], "number"
            elif pool and txt:
                best = max(((_score(txt, x["name"]), x) for x in pool),
                           default=(0.0, None), key=lambda t: t[0])
                name, how = ((best[1]["name"], "name-only") if best[0] >= 0.6
                             else (txt, "text-as-is"))
            else:
                name, how = txt, "no-roster" if not pool else "text-as-is"
            out.append({"number": num, "name": name, "position": p.get("position")})
            hows.append(how)
        return out, hows

    players, ph = rows("players")
    subs, sh = rows("substitutes")
    team = roster["teams"][side]["name"] if (roster and side) else vlm.get("team_name")
    manager = vlm.get("manager_name")
    if roster and side:
        coach = roster["teams"][side].get("coach")
        if coach and manager and _score(manager, coach) >= 0.55:
            manager = coach
    meta = {"team_side": side, "player_hows": ph, "substitute_hows": sh}
    if wrong_match:
        meta["warning"] = (f"printed team {vlm.get('team_name')!r} matches neither "
                           f"roster team - roster NOT applied, check the match")
    return {"team_name": team, "formation": vlm.get("formation"),
            "players": players, "substitutes": subs,
            "manager_name": manager, "_cache": meta}


# --------------------------------------------------------------------- scoreboard
_CLOCK = re.compile(r"^(\d{1,3})[:.](\d{2})$")


def period_from_clock(match_time) -> str | None:
    m = _CLOCK.match(str(match_time or "").strip())
    if not m:
        return None
    mins = int(m.group(1))
    return ("1H" if mins <= 45 else "2H" if mins <= 90
            else "ET1" if mins <= 105 else "ET2")


def resolve_main_scoreboard(vlm: dict, roster: dict | None) -> dict:
    out = {k: vlm.get(k) for k in ("home_team", "away_team", "home_score",
                                   "away_score", "match_time", "added_time")}
    if roster:
        for key in ("home_team", "away_team"):
            s = team_side(roster, out.get(key))
            if s:
                out[key] = roster["teams"][s]["name"]
    out["period"] = period_from_clock(out.get("match_time"))
    out["_cache"] = {"period": "derived-from-clock"}
    return out


RESOLVERS = {"card_event": resolve_card_event,
             "substitution": resolve_substitution,
             "lineup": resolve_lineup,
             "main_scoreboard": resolve_main_scoreboard}
