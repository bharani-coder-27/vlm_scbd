"""
Demo server: upload a video in the browser, pick a model, watch results stream in.

    python server.py                    # then open http://localhost:8000

Frames are sampled, YOLO labels each graphic, the crop goes to the selected MLX
model, and the roster cache turns shirt numbers into names. Results are pushed over
SSE as they are produced, so the page fills in while the video is still processing.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from detector import GraphicDetector, pick_device        # noqa: E402
from mlx_backend import BACKEND                          # noqa: E402
from pipeline import LABELS, process_video               # noqa: E402

ROSTERS = ROOT / "rosters"
UPLOADS = Path(tempfile.gettempdir()) / "graphics_demo"
UPLOADS.mkdir(exist_ok=True)

app = FastAPI(title="Broadcast Graphics Demo")
_detector: GraphicDetector | None = None


def detector() -> GraphicDetector:
    global _detector
    if _detector is None:
        _detector = GraphicDetector()
    return _detector


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/config")
def config():
    try:
        d = detector()
        yolo = {"weights": Path(d.weights).name, "device": d.device,
                "classes": list(d.names.values()), "ok": True}
    except Exception as e:                               # noqa: BLE001
        yolo = {"ok": False, "error": str(e)}
    return {"models": BACKEND.available(),
            "active_model": BACKEND.key,
            "rosters": sorted(p.stem for p in ROSTERS.glob("*.json")),
            "labels": LABELS,
            "yolo": yolo,
            "device": pick_device()}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "no filename")
    job = uuid.uuid4().hex[:12]
    dest = UPLOADS / f"{job}{Path(file.filename).suffix or '.mp4'}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    size = dest.stat().st_size
    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "empty upload")
    return {"job": job, "path": str(dest), "name": file.filename,
            "size_mb": round(size / 1024 / 1024, 1)}


@app.get("/api/process")
def process(path: str, model: str = "qwen3vl-4b", roster: str | None = None,
            fps: float = 1.0, conf: float | None = None,
            labels: str | None = None, merge: bool = True):
    """SSE stream of pipeline events."""
    src = Path(path)
    if not src.exists():
        raise HTTPException(404, f"no such upload: {path}")
    roster_path = None
    if roster:
        rp = ROSTERS / f"{roster}.json"
        if not rp.exists():
            raise HTTPException(404, f"no roster {roster!r}")
        roster_path = str(rp)
    want = [s.strip() for s in labels.split(",")] if labels else None

    def stream():
        try:
            for ev in process_video(str(src), detector(), model,
                                    roster_path=roster_path, fps=fps, conf=conf,
                                    labels=want, merge=merge):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as e:                           # noqa: BLE001
            import traceback
            traceback.print_exc()
            yield ("data: " + json.dumps(
                {"type": "error", "message": f"{type(e).__name__}: {e}"}) + "\n\n")

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


if __name__ == "__main__":
    import uvicorn
    print(f"device: {pick_device()}   rosters: "
          f"{[p.stem for p in ROSTERS.glob('*.json')]}")
    print("open http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
