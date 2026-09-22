"""
PaddleOCR scoreboard extractor — ported from D:/Extractor-SB/1_annotate_db/paddle_extract.py.

    general TEXT DETECTION -> TEXT RECOGNITION -> SEMANTIC PARSER

Layout-agnostic: it detects *all* text in the scoreboard crop and assigns each token to a
field by content pattern + geometry, so it works on layouts it has never seen. An optional
team list (from the match roster) makes team detection robust via fuzzy match.

~100 ms/crop vs ~4.8 s for the VLM, and more accurate on this field. Selected with
settings.scoreboard_backend = "paddleocr".
"""
from __future__ import annotations

import difflib
import re
import threading
import time
import unicodedata

import numpy as np

# --- field patterns ---------------------------------------------------------
CLOCK_RE = re.compile(r"^\d{1,2}[:.]\d{2}$")            # 22:25, 00:55
ET_RE = re.compile(r"^\+\d{1,2}$")                      # +5
DIGIT_LOOKALIKE = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "|": "1",
                   "Z": "2", "S": "5", "B": "8", "G": "6"}


def _to_digit(ch: str):
    ch = ch.upper()
    return ch if ch.isdigit() else DIGIT_LOOKALIKE.get(ch)


def score_value(text: str):
    """('pair', h, a) or ('one', v) if `text` is a plausible score, tolerating
    digit-lookalike misreads (O->0); else None."""
    t = (text or "").upper().replace(" ", "")
    if not t:
        return None
    m = re.search(r"(.)\s*[-:+.,/|]\s*(.)", t)          # "0-0", "3.3"
    if m:
        a, b = _to_digit(m.group(1)), _to_digit(m.group(2))
        if a is not None and b is not None:
            return ("pair", a, b)
    t = t.strip("-:+.,/|")
    if 1 <= len(t) <= 2:
        ds = [_to_digit(c) for c in t]
        if all(d is not None for d in ds):
            return ("one", "".join(ds))
    return None


def norm(s: str) -> str:
    """Uppercase, strip accents & non-alnum (WPL -> WPL)."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def match_team(text: str, teams: list[str], thresh: float = 0.55):
    """Fuzzy-match an OCR token to the closest known team."""
    if not teams:
        return None, 0.0
    key, best, best_r = norm(text), None, 0.0
    if not key:
        return None, 0.0
    for t in teams:
        r = difflib.SequenceMatcher(None, key, norm(t)).ratio()
        if r > best_r:
            best_r, best = r, t
    return (best, best_r) if best_r >= thresh else (None, best_r)


# --- semantic parser --------------------------------------------------------
def choose_clocks(clocks: list[dict], all_tokens: list[dict]):
    """main_clock = top-most clock at a horizontal end; added_clock = the one below it
    (the stoppage counter that starts at 00:00 when the main clock freezes)."""
    if not clocks:
        return None, None
    minx = min(t["box"][0] for t in all_tokens)
    maxx = max(t["box"][2] for t in all_tokens)

    def key(t):
        d_end = min(t["box"][0] - minx, maxx - t["box"][2])
        return (round(t["yc"]), d_end)

    ordered = sorted(clocks, key=key)
    return ordered[0], (ordered[1] if len(ordered) > 1 else None)


def choose_extra(extras: list[dict], anchor):
    """Extra time belongs to its clock: pick the candidate nearest the anchor."""
    if not extras:
        return None
    if anchor is not None:
        ax, ay = anchor["xc"], anchor["yc"]
        return min(extras, key=lambda e: (abs(e["yc"] - ay), abs(e["xc"] - ax)))["text"]
    return max(extras, key=lambda e: e["conf"])["text"]


def split_tokens(tokens: list[dict]) -> list[dict]:
    """One detected region can hold several fields ('0:38 +5'); split on whitespace
    and spread the parts across the box left->right."""
    out = []
    for t in tokens:
        parts = (t["text"] or "").split()
        if len(parts) <= 1:
            out.append(t)
            continue
        x1, y1, x2, y2 = t["box"]
        w = (x2 - x1) / len(parts)
        for k, part in enumerate(parts):
            nx1 = x1 + k * w
            out.append({**t, "text": part, "xc": nx1 + w / 2,
                        "box": [int(nx1), y1, int(nx1 + w), y2]})
    return out


def parse(tokens: list[dict], teams: list[str]) -> dict:
    toks = sorted(split_tokens(tokens), key=lambda t: t["xc"])
    clocks, extras, team_hits = [], [], []

    # pass 1: clocks, extra time, teams (score tokens left for pass 2)
    for t in toks:
        s = (t["text"] or "").upper().replace(" ", "")
        if not s:
            continue
        if ET_RE.match(s):
            extras.append(t); t["_role"] = "et"
        elif CLOCK_RE.match(s):
            clocks.append(t); t["_role"] = "clock"
        else:
            canon, r = match_team(t["text"], teams)
            if canon:
                team_hits.append((t, canon, r)); t["_role"] = "team"
            elif (not teams and re.search(r"[A-Za-z]", t["text"])
                  and len(norm(t["text"])) >= 2 and score_value(t["text"]) is None):
                team_hits.append((t, t["text"], 0.0)); t["_role"] = "team"

    # teams: two best distinct matches; detect HORIZONTAL vs VERTICAL layout
    best_by_canon = {}
    for tok, canon, r in team_hits:
        if canon not in best_by_canon or r > best_by_canon[canon][2]:
            best_by_canon[canon] = (tok, canon, r)
    chosen = sorted(best_by_canon.values(), key=lambda x: -x[2])[:2]
    team_ids = {id(c[0]) for c in chosen}

    home = away = home_tok = away_tok = None
    vertical = False
    if len(chosen) >= 2:
        a, b = chosen[0][0], chosen[1][0]
        vertical = abs(a["yc"] - b["yc"]) > abs(a["xc"] - b["xc"])
        order = (lambda c: c[0]["yc"]) if vertical else (lambda c: c[0]["xc"])
        top2 = sorted(chosen, key=order)
        home_tok, home = top2[0][0], top2[0][1]
        away_tok, away = top2[1][0], top2[1][1]
    elif len(chosen) == 1:
        home_tok, home = chosen[0][0], chosen[0][1]

    # pass 2: score-like tokens (O->0), splitting merged/combined digits
    gap = abs(home_tok["xc"] - away_tok["xc"]) if (home_tok and away_tok) else 0
    score_toks = []
    for t in toks:
        if id(t) in team_ids or t.get("_role") in ("clock", "et"):
            continue
        sv = score_value(t["text"])
        if not sv:
            continue
        x1, _, x2, _ = t["box"]
        q = (x2 - x1) / 4
        if sv[0] == "pair":
            score_toks.append((x1 + q, t["yc"], sv[1]))
            score_toks.append((x2 - q, t["yc"], sv[2]))
        elif len(sv[1]) == 2 and not vertical and gap and (x2 - x1) > 0.30 * gap:
            score_toks.append((x1 + q, t["yc"], sv[1][0]))
            score_toks.append((x2 - q, t["yc"], sv[1][1]))
        else:
            score_toks.append((t["xc"], t["yc"], sv[1]))

    # assign home/away scores by geometry
    hs = aw = None
    if home_tok and away_tok:
        if vertical:
            row_h = abs(home_tok["yc"] - away_tok["yc"]) / 2

            def by_row(team):
                c = [s for s in score_toks if abs(s[1] - team["yc"]) < row_h] or score_toks
                return min(c, key=lambda s: abs(s[1] - team["yc"]))[2] if c else None

            hs, aw = by_row(home_tok), by_row(away_tok)
        else:
            lo, hi = sorted((home_tok["xc"], away_tok["xc"]))
            tol = 0.15 * (hi - lo)
            inner = sorted([s for s in score_toks if lo - tol <= s[0] <= hi + tol],
                           key=lambda s: s[0])
            if len(inner) >= 2:
                hs, aw = inner[0][2], inner[-1][2]
            elif len(inner) == 1:
                s = inner[0]
                if abs(s[0] - away_tok["xc"]) < abs(s[0] - home_tok["xc"]):
                    aw = s[2]
                else:
                    hs = s[2]
    elif home_tok and score_toks:
        hs = min(score_toks, key=lambda s: abs(s[0] - home_tok["xc"]))[2]

    main_tok, added_tok = choose_clocks(clocks, toks)
    anchor = added_tok or main_tok
    return {
        "main_clock": main_tok["text"] if main_tok else None,
        "added_clock": added_tok["text"] if added_tok else None,
        "extra_time": choose_extra(extras, anchor),
        "teams": {"home": home, "away": away},
        "score": {"home": int(hs) if hs is not None else None,
                  "away": int(aw) if aw is not None else None},
        "raw_tokens": [{"text": t["text"], "conf": t["conf"]} for t in toks],
    }




# ============================================================================
# OCR engines. The parser above only needs a list of
#     {"text","conf","box":[x1,y1,x2,y2],"xc","yc"}
# so any engine works. Tried in order; the first that imports wins.
#
#   apple  - macOS Vision framework via pyobjc. Built into the OS, no model to
#            download, ~50-120 ms on a scoreboard crop. Best choice on a Mac.
#   rapid  - RapidOCR (onnxruntime). Cross-platform, ARM-friendly, ~150-300 ms.
#   paddle - PaddleOCR. What the Windows pipeline uses (774 ms measured), but
#            PaddlePaddle has no reliable Apple-Silicon wheel.
# ============================================================================
import threading  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402


class AppleVisionOCR:
    """macOS Vision text recognition. pip install pyobjc-framework-Vision"""

    name = "apple"

    def __init__(self):
        import Quartz  # noqa: F401
        import Vision
        from Foundation import NSData
        self._Vision, self._NSData = Vision, NSData
        import cv2
        self._cv2 = cv2

    def tokens(self, im: np.ndarray) -> list[dict]:
        import Quartz
        H, W = im.shape[:2]
        ok, buf = self._cv2.imencode(".png", im)
        if not ok:
            return []
        data = self._NSData.dataWithBytes_length_(buf.tobytes(), len(buf))
        src = Quartz.CGImageSourceCreateWithData(data, None)
        cg = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
        req = self._Vision.VNRecognizeTextRequest.alloc().init()
        req.setRecognitionLevel_(0)              # 0 = accurate
        req.setUsesLanguageCorrection_(False)    # scoreboards are codes, not prose
        handler = self._Vision.VNImageRequestHandler.alloc()             .initWithCGImage_options_(cg, None)
        handler.performRequests_error_([req], None)
        out = []
        for obs in (req.results() or []):
            cand = obs.topCandidates_(1)
            if not cand:
                continue
            txt = cand[0].string()
            bb = obs.boundingBox()              # normalised, origin bottom-left
            x1 = bb.origin.x * W
            x2 = (bb.origin.x + bb.size.width) * W
            y1 = (1 - bb.origin.y - bb.size.height) * H
            y2 = (1 - bb.origin.y) * H
            out.append({"text": (txt or "").strip(),
                        "conf": round(float(cand[0].confidence()), 3),
                        "box": [int(x1), int(y1), int(x2), int(y2)],
                        "xc": (x1 + x2) / 2, "yc": (y1 + y2) / 2})
        return out


class RapidOCREngine:
    """RapidOCR / onnxruntime. pip install rapidocr-onnxruntime"""

    name = "rapid"

    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR
        self._ocr = RapidOCR()

    def tokens(self, im: np.ndarray) -> list[dict]:
        res, _ = self._ocr(im)
        out = []
        for box, text, conf in (res or []):
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            out.append({"text": (text or "").strip(), "conf": round(float(conf), 3),
                        "box": [int(x1), int(y1), int(x2), int(y2)],
                        "xc": (x1 + x2) / 2, "yc": (y1 + y2) / 2})
        return out


class PaddleEngine:
    """PaddleOCR - what the Windows pipeline uses."""

    name = "paddle"

    def __init__(self, device: str = "cpu"):
        from paddleocr import TextDetection, TextRecognition
        self._det = TextDetection(device=device, enable_mkldnn=False)
        self._rec = TextRecognition(device=device)

    def tokens(self, im: np.ndarray) -> list[dict]:
        res = self._det.predict(im)[0]
        out = []
        for p in res["dt_polys"]:
            xs = [pt[0] for pt in p]
            ys = [pt[1] for pt in p]
            x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
            patch = im[max(0, y1):y2, max(0, x1):x2]
            if patch.size == 0:
                continue
            r = self._rec.predict(patch)[0]
            out.append({"text": (r.get("rec_text") or "").strip(),
                        "conf": round(float(r.get("rec_score", 0.0)), 3),
                        "box": [x1, y1, x2, y2],
                        "xc": (x1 + x2) / 2, "yc": (y1 + y2) / 2})
        return out


ENGINES = {"apple": AppleVisionOCR, "rapid": RapidOCREngine, "paddle": PaddleEngine}
PREFERENCE = ["apple", "rapid", "paddle"]


class ScoreboardOCR:
    """Whichever OCR engine is available + the semantic parser."""

    def __init__(self, engine: str | None = None):
        order = [engine] if engine else PREFERENCE
        errors = []
        self.engine = None
        for name in order:
            try:
                self.engine = ENGINES[name]()
                break
            except Exception as e:                       # noqa: BLE001
                errors.append(f"{name}: {type(e).__name__}")
        if self.engine is None:
            raise RuntimeError("no OCR engine available -> " + "; ".join(errors))
        self.name = self.engine.name
        self._lock = threading.Lock()

    def extract(self, im: np.ndarray, teams: list[str] | None = None):
        """-> (parsed, inference_ms) in the parser's native shape."""
        t0 = time.perf_counter()
        with self._lock:
            toks = self.engine.tokens(im)
        return parse(toks, teams or []), (time.perf_counter() - t0) * 1000


def to_scoreboard_schema(p: dict) -> dict:
    """Map the parser output onto the pipeline's main_scoreboard schema."""
    et = p.get("extra_time")
    added = None
    if et:
        m = re.search(r"\d+", str(et))
        if m:
            added = int(m.group(0))
    return {"home_team": p["teams"]["home"], "away_team": p["teams"]["away"],
            "home_score": p["score"]["home"], "away_score": p["score"]["away"],
            "match_time": p.get("main_clock"), "added_time": added,
            "added_clock": p.get("added_clock")}
