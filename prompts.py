"""
Prompt set for the Mac demo - the hybrid that benchmarked best.

  card_event / substitution / main_scoreboard : v4, numbers only.
      The model reads shirt numbers, card colour, arrow direction and a one-word
      `crest`. Player and team names come from the roster cache, not the model.

  lineup : v3, keeps `name_text`.
      Measured: dropping names from the line-up schema made the model lose its
      "one row = one player" anchor - correct XI counts fell from 7/9 to 3/9 and
      9 shirt numbers came back null. The name is only a row anchor; resolution
      is still number-first.
"""
from __future__ import annotations

CARD_EVENT_PROMPT = """
Read this football (soccer) card-event banner. Report ONLY the shirt number and the card colour.

The banner shows a small flag/crest, the player's SHIRT NUMBER in a coloured pill, the
player's name, and a solid card icon.

Return ONLY a single-line minified JSON object with EXACTLY these keys:
- "number": the shirt number as an integer (the digits in the coloured pill). null if unreadable.
- "card_type": decided ONLY by the colour of the card icon:
    one yellow/amber icon                      -> "yellow"
    one solid red icon                         -> "red"
    a yellow AND a red icon together, or two yellows -> "second_yellow"
  Never null, never "unknown".
- "crest": the country or club shown on the small flag/crest, as ONE short string
  (e.g. "Uruguay", "Spain"). null if you cannot tell. This is the only thing that
  says WHICH team, so give your best single guess.

Do NOT report the player name. Keep "crest" to one or two words.
JSON only. No prose, no markdown.

Example: {"number":7,"card_type":"yellow","crest":"Uruguay"}
""".strip()

SUBSTITUTION_PROMPT = """
Read this football (soccer) substitution banner. Report ONLY shirt numbers and direction.

The banner lists ONE, TWO or THREE players. Each has a SHIRT NUMBER in a coloured pill.
There is a direction arrow / marker, usually green or red:
  GREEN  -> those players are coming ON   -> direction "in"
  RED    -> those players are going OFF   -> direction "out"
The arrow applies to EVERY listed player, unless a row has its own coloured marker,
in which case use that row's own colour.

Return ONLY a single-line minified JSON object with EXACTLY these keys:
- "players": a JSON array, one object per listed player, top to bottom:
    {"number": <int or null>, "direction": "in" | "out" | null}
- "arrow": the colour of the main arrow: "green", "red", "both", or null.
- "crest": the country or club shown on the small flag/crest, as ONE short string
  (e.g. "Uruguay", "Spain"). null if you cannot tell. This is the only thing that
  says WHICH team, so give your best single guess.

Do NOT report player names. Keep "crest" to one or two words.
Read the NUMBERS carefully; they are the only thing that matters here.
If the banner lists no players, return {"players":[],"arrow":null}.
JSON only. No prose, no markdown.

Examples:
{"players":[{"number":21,"direction":"in"}],"arrow":"green","crest":"Uruguay"}
{"players":[{"number":23,"direction":"out"},{"number":6,"direction":"out"}],"arrow":"red","crest":"Uruguay"}
""".strip()

MAIN_SCOREBOARD_PROMPT = """
Read this football (soccer) scoreboard bug. Report ONLY what is visibly printed.

It is a small persistent overlay showing the two team codes, the two scores, and the clock.
Left/top = home, right/bottom = away.

Return ONLY a single-line minified JSON object with EXACTLY these keys:
- "home_team": the LEFT (or TOP) team's code exactly as printed.
- "away_team": the RIGHT (or BOTTOM) team's code exactly as printed.
- "home_score": the LEFT (or TOP) score as an integer.
- "away_score": the RIGHT (or BOTTOM) score as an integer.
- "match_time": the FULL clock exactly as shown, minutes AND seconds, e.g. "19:53",
  "45:00", "90+2". NEVER return only the seconds. null if no clock is visible.
- "added_time": added/stoppage minutes if shown separately (integer); else null.

The period is derived from the clock afterwards - do not report it.
JSON only. No prose, no markdown.

Example: {"home_team":"URU","away_team":"ESP","home_score":0,"away_score":1,"match_time":"45:00","added_time":8}
""".strip()

LINEUP_PROMPT = """
Read this football (soccer) starting line-up graphic. Report ONLY what is visibly printed.

It shows ONE team: a big team name + flag at the top, a formation string (e.g. "4-4-2" or
"4-1-2-3"), the STARTING XI, and a head coach by a "HEAD COACH" / "MANAGER" label.

The STARTING XI appears in one of two layouts - handle whichever you see:
  (A) a vertical LIST, each row = shirt number + player name; or
  (B) a PITCH MAP with a player photo/shirt per position, each labelled with a shirt
      number and a name underneath.
Some graphics ALSO have a "SUBSTITUTES" / "SUBS" / "BENCH" panel (usually down one side)
listing more number + name pairs. If that panel is present, report it; if it is absent,
return an empty list.

Return ONLY a single-line minified JSON object with EXACTLY these keys:
- "team_name": the big team name printed at the top.
- "formation": the formation string exactly as printed; if none is printed, null.
- "players": a JSON array for the STARTING XI only (layout A rows, or layout B pitch labels):
    {"number": <int or null>, "name_text": "<verbatim name or null>", "position": <"GK"|"DF"|"MF"|"FW"|null>}
    Read the NUMBER carefully - it matters most. In layout B order them back-to-front
    (goalkeeper first). position: only if clear from where the player sits on the pitch, else null.
- "substitutes": a JSON array in the SAME object shape, for the substitutes/bench panel.
    Use [] if this graphic has no such panel.
- "manager_name": the name printed by "HEAD COACH" / "MANAGER"; null if not shown.

Rules:
- Return exactly what you can see. If only 9 starters are readable, return 9 - never invent
  players and never use squad knowledge. Only what is in THIS image. Unreadable -> null.
- Do not put a player in both "players" and "substitutes".
- JSON only. No prose, no markdown.

Example:
{"team_name":"URUGUAY","formation":"4-1-2-3","players":[{"number":23,"name_text":"F. MUSLERA","position":"GK"},{"number":16,"name_text":"M. OLIVERA","position":"DF"}],"substitutes":[{"number":1,"name_text":"SERGIO ROCHET","position":"GK"}],"manager_name":"MARCELO BIELSA"}
""".strip()

PROMPTS: dict[str, str] = {
    "card_event": CARD_EVENT_PROMPT,
    "substitution": SUBSTITUTION_PROMPT,
    "lineup": LINEUP_PROMPT,
    "main_scoreboard": MAIN_SCOREBOARD_PROMPT,
}

MAX_NEW_TOKENS: dict[str, int] = {
    "card_event": 96,
    "substitution": 160,
    "lineup": 900,
    "main_scoreboard": 160,
}
