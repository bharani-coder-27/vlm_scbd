"""
MLX model backend for Apple Silicon. Loads one VLM at a time and caches it, so
switching models in the UI only reloads when the selection actually changes.

Both models take the same call shape, so the pipeline does not care which is active.
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

MODELS: dict[str, dict] = {
    "qwen3vl-4b": {
        "repo": "mlx-community/Qwen3-VL-4B-Instruct-4bit",
        "label": "Qwen3-VL-4B (MLX 4-bit)",
        "note": "general VLM; strongest on line-ups",
    },
    "nuextract3": {
        "repo": "numind/NuExtract3-mlx-4bits",
        "label": "NuExtract3 (MLX 4-bit)",
        "note": "extraction-tuned; ~5x faster decode on CUDA/GGUF",
    },
}

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str):
    """First {...} in the reply, tolerating ``` fences and trailing prose."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I).strip()
    m = _JSON_RE.search(t)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:                                    # noqa: BLE001
        return None


class MLXBackend:
    """One loaded model at a time, guarded by a lock (MLX is not thread-safe here)."""

    def __init__(self):
        self.key: str | None = None
        self.repo: str | None = None
        self._model = None
        self._processor = None
        self._config = None
        self._lock = threading.Lock()
        self.load_seconds: float | None = None

    def available(self) -> list[dict]:
        return [{"key": k, **v} for k, v in MODELS.items()]

    def ensure(self, key: str) -> None:
        """Load `key` if it is not already the active model."""
        if key not in MODELS:
            raise ValueError(f"unknown model {key!r}; expected one of {list(MODELS)}")
        with self._lock:
            if self.key == key and self._model is not None:
                return
            from mlx_vlm import load
            from mlx_vlm.utils import load_config

            repo = MODELS[key]["repo"]
            # drop the previous model first so both never sit in unified memory
            self._model = self._processor = self._config = None
            try:
                import gc

                import mlx.core as mx
                gc.collect()
                mx.clear_cache()
            except Exception:                            # noqa: BLE001
                pass
            t0 = time.perf_counter()
            self._model, self._processor = load(repo)
            self._config = load_config(repo)
            self.load_seconds = round(time.perf_counter() - t0, 1)
            self.key, self.repo = key, repo

    def generate(self, image: "Image.Image | np.ndarray | str | Path",
                 prompt: str, max_tokens: int = 320) -> tuple[str, float, int | None]:
        """-> (text, elapsed_ms, generated_tokens|None)"""
        from mlx_vlm import generate as mlx_generate
        from mlx_vlm.prompt_utils import apply_chat_template

        if isinstance(image, np.ndarray):                # BGR (cv2) -> RGB PIL
            image = Image.fromarray(image[:, :, ::-1])
        if isinstance(image, Image.Image):
            tmp = Path(".mlx_tmp.jpg")
            image.convert("RGB").save(tmp, quality=92)
            image = tmp                                  # mlx-vlm wants a path
        with self._lock:
            formatted = apply_chat_template(self._processor, self._config,
                                            prompt, num_images=1)
            t0 = time.perf_counter()
            try:
                out = mlx_generate(self._model, self._processor, formatted,
                                   [str(image)], max_tokens=max_tokens,
                                   temperature=0.0, verbose=False)
            except TypeError:                            # older mlx-vlm signature
                out = mlx_generate(self._model, self._processor, formatted,
                                   [str(image)], max_tokens=max_tokens,
                                   temp=0.0, verbose=False)
            dt = (time.perf_counter() - t0) * 1000

        ntok = None
        if isinstance(out, str):
            text = out
        elif isinstance(out, tuple):
            text = out[0]
        else:
            text = getattr(out, "text", str(out))
            ntok = getattr(out, "generation_tokens", None)
        return text, dt, ntok


BACKEND = MLXBackend()
