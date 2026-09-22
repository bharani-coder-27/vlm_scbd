"""YOLO graphic detector for the Mac demo (Apple Silicon: MPS, falling back to CPU)."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

# These weights are trained on the 4 broadcast graphic classes and degrade sharply
# above 640 - on the test set line-up confidence fell 0.98 -> 0.22 at imgsz 960.
DEFAULT_WEIGHTS = Path(__file__).parent / "weights" / "best.pt"
IMGSZ = 640
CONF = 0.25
IOU = 0.5


def pick_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:                                    # noqa: BLE001
        pass
    return "cpu"


class GraphicDetector:
    def __init__(self, weights: str | Path | None = None, device: str | None = None):
        from ultralytics import YOLO

        self.weights = str(weights or DEFAULT_WEIGHTS)
        if not Path(self.weights).exists():
            raise FileNotFoundError(
                f"YOLO weights not found at {self.weights}. Copy best.pt into "
                f"mac_demo/weights/ (see README).")
        self.device = device or pick_device()
        self._model = YOLO(self.weights)
        self.names: dict[int, str] = dict(self._model.names)
        self._lock = threading.Lock()

    def detect(self, image: np.ndarray, conf: float | None = None,
               labels: list[str] | None = None) -> tuple[list[dict], float]:
        """-> (detections sorted by confidence, inference_ms). One box per label."""
        t0 = time.perf_counter()
        with self._lock:                                 # ultralytics predict isn't thread-safe
            res = self._model.predict(image, conf=conf if conf is not None else CONF,
                                      iou=IOU, imgsz=IMGSZ, device=self.device,
                                      verbose=False)[0]
        dt = (time.perf_counter() - t0) * 1000

        out: list[dict] = []
        for b in res.boxes:
            name = self.names.get(int(b.cls.item()), str(int(b.cls.item())))
            if labels and name not in labels:
                continue
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            out.append({"label": name, "confidence": round(float(b.conf.item()), 4),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)]})
        out.sort(key=lambda d: -d["confidence"])

        seen, dedup = set(), []
        for d in out:                                    # keep the best box per label
            if d["label"] in seen:
                continue
            seen.add(d["label"])
            dedup.append(d)
        return dedup, dt
