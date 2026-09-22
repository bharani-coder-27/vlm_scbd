# Broadcast Graphics Extraction — Mac demo

Upload a video in the browser → frames are sampled → YOLO labels each graphic →
the crop goes to the selected MLX model → the roster cache turns shirt numbers into
names → the frame and its final JSON appear side by side, streaming in as it runs.

```
video ─▶ sample @1fps ─▶ YOLO ─┬─ main_scoreboard ─┐
                               ├─ lineup           ├─▶ MLX VLM ─▶ roster cache ─▶ JSON
                               ├─ substitution     │
                               └─ card_event       ┘
```

Everything runs locally. No API keys, no network.

---

## 1. Copy this folder to the Mac

The whole thing is ~6 MB and self-contained — the YOLO weights are already in
`weights/best.pt` and the rosters in `rosters/`.

```bash
scp -r mac_demo/ you@mac:~/graphics-demo
```

## 2. Install

```bash
cd ~/graphics-demo
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Check MLX has the GPU (must print `Device(gpu, 0)`, and `arm64` not `x86_64`):

```bash
python -c "import mlx.core as mx, platform; print(mx.default_device(), platform.machine())"
```

If it says `x86_64` you are in Rosetta — reinstall Python as native arm64.

## 3. Models

Both are already installed on your machine. To pre-pull explicitly:

```bash
hf download mlx-community/Qwen3-VL-4B-Instruct-4bit
hf download numind/NuExtract3-mlx-4bits
```

Only one is resident at a time — switching in the UI unloads the previous model
first, so a 16 GB Mac never holds both.

## 4. Run

```bash
python server.py
```

Open <http://localhost:8000>, drop a video, pick a model and a roster, press
**Process video**.

---

## What the UI shows

Each detected graphic becomes a card: **the exact crop the model saw** on the left,
**the final JSON** on the right. "show raw model output" reveals what the model
returned *before* the cache filled in names — that side-by-side is the point of the
demo.

Blue-bordered **merged** cards appear when a graphic leaves the screen: one event
per appearance, combining every read of it.

Sidebar counters: frames read, detections, extractions, and how many detections the
event trigger skipped (this number should be large — see below).

---

## Design notes

**The event trigger is presence-based, not pixel-based.** Pixel SSIM on a detection
crop does not work here: the panels are semi-transparent over live footage and the
YOLO box jitters, so two crops of an *unchanged* graphic score only 0.55–0.69 SSIM.
Keying on "is this label on screen" instead cut line-up reads 54 → 9 on the test
clip; the backoff (interval doubles while the answer repeats, capped at 20 s) takes
it to ~5. The cap matters: Uruguay and Spain appear in one continuous `lineup`
presence run, and an uncapped backoff reads the second team ~26 s late.

**The roster cache does the identifying, not the model.** The model reads shirt
numbers, card colour, arrow direction and a one-word `crest`; the cache maps the
number to a player. This removed every OCR-name error (the broadcast font renders Y
as V — `LAMINE VAMAL`, `DAVID RAVA`, `AVMERIC LAPORTE`) and every wrong-country
guess.

**Line-up keeps `name_text`; the banners do not.** Measured: dropping names from the
line-up schema made the model lose its "one row = one player" anchor — correct XI
counts fell from 7/9 to 3/9 and 9 shirt numbers came back `null`. The name is only a
row anchor; resolution is still number-first.

**`crest` exists because numbers are not unique.** Uruguay and Spain share nearly
every shirt number (#8 is Valverde *and* Fabián Ruiz). Line-up graphics print the
team name so they are fine; banners print only a flag. When `crest` is missing or
wrong the resolver marks the result `ambiguous` and returns **both** candidates
rather than guessing — look for `_cache.reason` in the raw JSON.

**Merging matters because the graphics animate.** The TVP line-up shows 11 starters
and no bench early, then the bench with two starters occluded. No single frame has
everything; the merge takes the most complete answer per field.

---

## Adding a match

Drop a `rosters/<match-id>.json` in, following the existing files:

```json
{
  "teams": {
    "home": {"name": "Uruguay", "short": "URU", "flag_colors": ["blue","white"], "coach": "Marcelo Bielsa"},
    "away": {"name": "Spain",   "short": "ESP", "flag_colors": ["red","yellow"], "coach": "Luis de la Fuente"}
  },
  "players": [
    {"team": "home", "number": 23, "name": "Fernando Muslera", "role": "starter", "position": "GK"}
  ]
}
```

It appears in the dropdown on reload. Without a roster the pipeline still runs —
shirt numbers come through and names are left `null`.

---

## Files

| file | role |
|---|---|
| `server.py` | FastAPI: upload, config, SSE stream |
| `pipeline.py` | decode → sample → YOLO → trigger → VLM → resolve → merge |
| `mlx_backend.py` | model registry, load/unload, generate |
| `detector.py` | YOLO wrapper (MPS, falls back to CPU) |
| `prompts.py` | hybrid prompt set — v4 banners, v3 line-up |
| `resolve.py` | roster cache + period-from-clock |
| `static/index.html` | the whole frontend, no build step |

## Tuning

| knob | where | note |
|---|---|---|
| sample rate | UI | 1 fps is plenty; a graphic is on screen for seconds |
| `RECHECK_S` | `pipeline.py` | how long a label may persist before re-reading |
| `max_recheck` | `EventTrigger` | backoff ceiling — keep ≤20 s or team changes are read late |
| `CROP_PAD` | `pipeline.py` | per-label crop padding |
| `IMGSZ` | `detector.py` | **leave at 640** — these weights collapse above it |

## Troubleshooting

| symptom | fix |
|---|---|
| `YOLO not loaded` in the sidebar | `weights/best.pt` missing — copy it back |
| `Device(cpu, 0)` from MLX | Python is running under Rosetta; reinstall native arm64 |
| First extraction very slow | model weights paging in; the second call is representative |
| `generate() TypeError` | mlx-vlm API drift — the code already tries `temperature=` then `temp=` |
| Everything `ambiguous` | no roster selected, or the wrong match picked |
