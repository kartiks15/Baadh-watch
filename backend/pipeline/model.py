"""Water segmentation inference (U-Net trained on Sen1Floods11, exported to ONNX)."""
from __future__ import annotations

import json

import numpy as np

from .config import MODELS, Thresholds


class WaterModel:
    def __init__(self, onnx_path=MODELS / "water_unet.onnx", meta_path=MODELS / "model_meta.json"):
        self.session = None
        self.meta = {}
        if onnx_path.exists() and meta_path.exists():
            import onnxruntime as ort

            self.meta = json.loads(meta_path.read_text())
            self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    @property
    def available(self) -> bool:
        return self.session is not None

    @property
    def name(self) -> str:
        if not self.available:
            return "threshold fallback (no model file)"
        return f"U-Net/{self.meta.get('encoder', '?')} trained on Sen1Floods11"

    def _scale(self, vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
        m = self.meta
        x = np.stack([vv, vh]).astype(np.float32)
        x = np.nan_to_num(x, nan=m["db_min"])
        x = (np.clip(x, m["db_min"], m["db_max"]) - m["db_min"]) / (m["db_max"] - m["db_min"])
        for c in range(2):
            x[c] = (x[c] - m["mean"][c]) / m["std"][c]
        return x

    def predict(self, vv: np.ndarray, vh: np.ndarray, th: Thresholds, tile: int = 512, pad: int = 64) -> np.ndarray:
        """P(water) for a sigma0 dB scene. Overlapping tiles; only the tile centres are kept."""
        if not self.available:
            # transparent fallback: logistic around fixed sigma0 thresholds
            z = 0.6 * (th.water_vv_db - vv) + 0.4 * (th.water_vh_db - vh)
            return np.nan_to_num(1 / (1 + np.exp(-z)), nan=0).astype(np.float32)
        x = self._scale(vv, vh)
        _, h, w = x.shape
        step = tile - 2 * pad
        H = int(np.ceil(h / step)) * step + 2 * pad
        W = int(np.ceil(w / step)) * step + 2 * pad
        xp = np.pad(x, ((0, 0), (pad, H - h - pad), (pad, W - w - pad)), mode="reflect")
        out = np.zeros((H, W), np.float32)
        for r in range(0, H - 2 * pad, step):
            for c in range(0, W - 2 * pad, step):
                logits = self.session.run(None, {"x": xp[None, :, r:r + tile, c:c + tile]})[0][0]
                e = np.exp(logits - logits.max(0))
                p = e[1] / e.sum(0)
                out[r + pad:r + pad + step, c + pad:c + pad + step] = p[pad:pad + step, pad:pad + step]
        return out[pad:pad + h, pad:pad + w]
