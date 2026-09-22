"""
Video -> frames -> YOLO -> crop -> VLM -> roster cache.

Yields events as it goes so the UI can render results while the video is still
being processed.

The event trigger is PRESENCE-based, not pixel-based. Pixel SSIM on a detection crop
does not work on this footage: the panels are semi-transparent over live video and the
YOLO box jitters, so consecutive crops of an UNCHANGED graphic score only ~0.55-0.69
SSIM. Keying on "is this label on screen" instead cut line-up reads 54 -> 9 on the
test clip, and the backoff below takes it to ~5.
"""
from __future__ import annotations

import base64
import time
from collections import Counter
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from mlx_backend import BACKEND, extract_json
from prompts import MAX_NEW_TOKENS, PROMPTS
from resolve import RESOLVERS, load_roster

LABELS = ["main_scoreboard", "lineup", "substitution", "card_event"]

# padding added around each detection, as a fraction of the box w/h
CROP_PAD = {"main_scoreboard": 0.08, "card_event": 0.10,
            "substitution": 0.08, "lineup": 0.02}

# how long a label may stay on screen before we re-read it (video-seconds)
RECHECK_S = {"main_scoreboard": 0.0, "card_event": 2.0,
             "substitution": 2.0, "lineup": 8.0}

MERGE_LABELS = {"lineup", "card_event", "substitution"}


def crop(img: np.ndarray, bbox, label: str) -> np.ndarray:
    H, W = img.shape[:2]
    x1, y1, x2, y2 = bbox
    pad = CROP_PAD.get(label, 0.05)
    bw, bh = x2 - x1, y2 - y1
    return img[max(0, int(y1 - bh * pad)):min(H, int(y2 + bh * pad)),
               max(0, int(x1 - bw * pad)):min(W, int(x2 + bw * pad))]


def to_data_uri(img: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


class EventTrigger:
    """Fire when a label appears; while it persists, re-fire on a backing-off
    interval that doubles whenever the answer comes back unchanged."""

    def __init__(self, gap_s=2.0, recheck=None, backoff=2.0, max_recheck=20.0):
        self.gap_s, self.backoff, self.max_recheck = gap_s, backoff, max_recheck
        self.base = dict(recheck or RECHECK_S)
        self._seen: dict[str, float] = {}
        self._fired: dict[str, float] = {}
        self._interval: dict[str, float] = {}
        self._hash: dict[str, str] = {}

    def run_ended(self, label: str, t: float) -> bool:
        last = self._seen.get(label)
        return last is not None and (t - last) > self.gap_s

    def should_fire(self, label: str, t: float) -> tuple[bool, str]:
        last = self._seen.get(label)
        self._seen[label] = t
        if last is None or (t - last) > self.gap_s:
            self._fired[label] = t
            self._interval[label] = self.base.get(label, 0.0)
            self._hash.pop(label, None)
            return True, "appeared"
        every = self._interval.get(label, self.base.get(label, 0.0))
        if every <= 0:
            self._fired[label] = t
            return True, "always"
        if t - self._fired.get(label, -1e9) >= every:
            self._fired[label] = t
            return True, "recheck"
        return False, "held"

    def note(self, label: str, data) -> None:
        """Feed the extraction back so the interval can back off while the answer
        repeats. Compares a COARSE signature, not the whole JSON: the line-up row
        count wobbles between reads (11/18/10 for the same graphic), so hashing the
        full object means the hash never repeats and the backoff never engages.
        The signature keeps the fields that actually identify the graphic."""
        import json as _json
        base = self.base.get(label, 0.0)
        if base <= 0:
            return
        if label == "lineup" and isinstance(data, dict):
            sig = {k: data.get(k) for k in ("team_name", "formation", "manager_name")}
        elif label == "substitution" and isinstance(data, dict):
            sig = {"team": data.get("team_name"), "dir": data.get("direction"),
                   "n": sorted(str(p.get("shirt_number"))
                               for p in data.get("players") or [])}
        else:
            sig = data
        h = _json.dumps(sig, sort_keys=True, default=str)
        prev = self._hash.get(label)
        self._hash[label] = h
        cur = self._interval.get(label, base)
        self._interval[label] = (min(cur * self.backoff, self.max_recheck)
                                 if prev == h else base)


def _mode(vals):
    import json as _json
    vals = [v for v in vals if v not in (None, "", [])]
    if not vals:
        return None
    c = Counter(_json.dumps(v, sort_keys=True, default=str) for v in vals)
    return _json.loads(c.most_common(1)[0][0])


def _vote_squad(reads: list[list[dict]], ratio: float = 0.5) -> list[dict]:
    """Majority-vote a player list by shirt number across the reads of one run.

    Taking the LONGEST read is wrong: the model sometimes over-generates rows (we
    measured XI counts of [11,18,10,11,11,11,11,11] for one graphic, and `max(len)`
    picked the 18). A real squad member appears in most reads; a hallucinated one
    appears in a single read.

    Empty reads are excluded from the denominator, because the bench panel genuinely
    only appears in some frames - counting the frames without it would delete the
    whole bench.
    """
    nonempty = [r for r in reads if r]
    if not nonempty:
        return []
    need = max(1, int(len(nonempty) * ratio))
    seen: dict = {}
    for r in nonempty:
        for p in r:
            key = p.get("number")
            if key is None:
                continue
            e = seen.setdefault(key, {"count": 0, "rows": []})
            e["count"] += 1
            e["rows"].append(p)
    out = []
    for key, e in seen.items():
        if e["count"] < need:
            continue
        row = dict(e["rows"][0])
        row["name"] = _mode([x.get("name") for x in e["rows"]]) or row.get("name")
        row["position"] = _mode([x.get("position") for x in e["rows"]])
        out.append(row)
    # keep the ordering of the read that agreed with the vote most closely
    order = max(nonempty, key=lambda r: sum(1 for p in r
                                            if p.get("number") in {o["number"] for o in out}))
    rank = {p.get("number"): i for i, p in enumerate(order)}
    out.sort(key=lambda p: rank.get(p["number"], 10_000))
    return out


def merge_run(label: str, items: list[dict]) -> dict:
    """One event per presence run. The graphics animate - the TVP line-up shows 11
    starters and no bench early, then the bench with two starters occluded - so
    combine the reads rather than trusting any single frame."""
    datas = [i["data"] for i in items if i.get("data")]
    if not datas:
        return {}
    if label == "lineup":
        return {"team_name": _mode([d.get("team_name") for d in datas]),
                "formation": _mode([d.get("formation") for d in datas]),
                "manager_name": _mode([d.get("manager_name") for d in datas]),
                "players": _vote_squad([d.get("players") or [] for d in datas]),
                "substitutes": _vote_squad([d.get("substitutes") or [] for d in datas]),
                "reads": len(datas)}
    if label == "card_event":
        return {k: _mode([d.get(k) for d in datas])
                for k in ("team_name", "player_name", "shirt_number",
                          "card_type", "minute")}
    if label == "substitution":
        seen: dict = {}
        for d in datas:
            for p in d.get("players") or []:
                key = p.get("shirt_number") or p.get("player_name")
                if key is None:
                    continue
                e = seen.setdefault(key, {"shirt_number": p.get("shirt_number"),
                                          "name": p.get("player_name"),
                                          "team": p.get("team_name"), "dirs": []})
                if p.get("direction"):
                    e["dirs"].append(p["direction"])
        players, ins, outs = [], [], []
        for e in seen.values():
            d = Counter(e["dirs"]).most_common(1)[0][0] if e["dirs"] else None
            players.append({"shirt_number": e["shirt_number"], "name": e["name"],
                            "team": e["team"], "direction": d,
                            "direction_votes": dict(Counter(e["dirs"]))})
            label_name = e["name"] or f"#{e['shirt_number']}"
            if d == "in":
                ins.append(label_name)
            elif d == "out":
                outs.append(label_name)
        return {"team_name": _mode([d.get("team_name") for d in datas]),
                "player_in": ins, "player_out": outs, "players": players}
    return datas[-1]


def process_video(video_path: str, detector, model_key: str,
                  roster_path: str | None = None, fps: float = 1.0,
                  conf: float | None = None, labels: list[str] | None = None,
                  merge: bool = True) -> Iterator[dict]:
    """Generator of dict events: start | frame | result | merged | done | error."""
    BACKEND.ensure(model_key)
    roster = load_roster(roster_path) if roster_path else None
    wanted = labels or LABELS

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        yield {"type": "error", "message": f"cannot open {video_path}"}
        return
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = n_frames / src_fps if src_fps else 0.0
    interval = max(1, int(round(src_fps / fps)))
    n_sampled = max(1, n_frames // interval)

    yield {"type": "start", "model": BACKEND.repo, "model_key": model_key,
           "load_seconds": BACKEND.load_seconds,
           "video": {"width": W, "height": H, "fps": round(src_fps, 2),
                     "frames": n_frames, "duration_s": round(duration, 1)},
           "sampling": {"target_fps": fps, "interval": interval,
                        "to_process": n_sampled},
           "roster": Path(roster_path).stem if roster_path else None,
           "labels": wanted}

    trigger = EventTrigger()
    open_runs: dict[str, list] = {}
    counts = Counter()
    t_wall0 = time.perf_counter()
    idx = -1
    done = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % interval:
            continue
        done += 1
        ts = idx / src_fps

        if merge:
            for lb in [k for k in list(open_runs) if trigger.run_ended(k, ts)]:
                items = open_runs.pop(lb)
                yield {"type": "merged", "label": lb, "t_start": items[0]["t"],
                       "t_end": items[-1]["t"], "reads": len(items),
                       "data": merge_run(lb, items)}

        dets, yolo_ms = detector.detect(frame, conf=conf, labels=wanted)
        counts["detections"] += len(dets)
        yield {"type": "frame", "i": done, "of": n_sampled, "t": round(ts, 2),
               "frame": idx, "detections": [d["label"] for d in dets],
               "yolo_ms": round(yolo_ms, 1),
               "elapsed_s": round(time.perf_counter() - t_wall0, 1)}

        for det in dets:
            label = det["label"]
            fire, why = trigger.should_fire(label, ts)
            if not fire:
                counts["skipped"] += 1
                continue
            patch = crop(frame, det["bbox"], label)
            if patch.size == 0:
                continue
            try:
                raw, ms, ntok = BACKEND.generate(patch, PROMPTS[label],
                                                 MAX_NEW_TOKENS.get(label, 320))
            except Exception as e:                        # noqa: BLE001
                yield {"type": "result", "t": round(ts, 2), "label": label,
                       "status": "error", "error": f"{type(e).__name__}: {e}",
                       "image": to_data_uri(patch)}
                continue
            parsed = extract_json(raw)
            counts["extractions"] += 1
            if parsed is None:
                counts["invalid_json"] += 1
                yield {"type": "result", "t": round(ts, 2), "label": label,
                       "status": "invalid_json", "raw": raw[:1500],
                       "latency_ms": round(ms, 1), "tokens": ntok,
                       "confidence": det["confidence"], "trigger": why,
                       "image": to_data_uri(patch)}
                continue
            data = RESOLVERS[label](parsed, roster)
            trigger.note(label, {k: v for k, v in data.items() if k != "_cache"})
            if merge and label in MERGE_LABELS:
                open_runs.setdefault(label, []).append({"t": round(ts, 2), "data": data})
            yield {"type": "result", "t": round(ts, 2), "label": label,
                   "status": "ok", "confidence": det["confidence"],
                   "latency_ms": round(ms, 1), "tokens": ntok, "trigger": why,
                   "model_output": parsed, "data": data,
                   "image": to_data_uri(patch)}

    cap.release()
    if merge:
        for lb, items in open_runs.items():
            yield {"type": "merged", "label": lb, "t_start": items[0]["t"],
                   "t_end": items[-1]["t"], "reads": len(items),
                   "data": merge_run(lb, items)}

    wall = time.perf_counter() - t_wall0
    yield {"type": "done", "wall_s": round(wall, 1),
           "frames_sampled": done, "counts": dict(counts),
           "realtime_factor": round(duration / max(0.001, wall), 2)}
