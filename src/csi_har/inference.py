"""Load a trained model and turn raw CSI into an activity prediction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from csi_har.config import ACTIVITIES, CLASS_NAMES, PreprocessConfig
from csi_har.data.reader import parse_csv_text
from csi_har.models import HARCNN
from csi_har.preprocess import preprocess_amplitude, preprocess_csi


@dataclass
class Prediction:
    activity: str
    activity_code: str
    confidence: float
    probabilities: dict[str, float]


def save_model(model: HARCNN, path: Path, cfg: PreprocessConfig, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_kwargs": model.hparams,
            "class_names": CLASS_NAMES,
            "target_length": cfg.target_length,
            "preprocess": asdict(cfg),
            "extra": extra or {},
        },
        path,
    )


class HARPredictor:
    def __init__(self, model_path: str | Path):
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        self.class_names: list[str] = checkpoint["class_names"]
        self.cfg = PreprocessConfig(**checkpoint["preprocess"])
        self.extra: dict = checkpoint.get("extra", {})
        self.model = HARCNN(**checkpoint.get("model_kwargs", {"n_classes": len(self.class_names)}))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    @property
    def needs_phase(self) -> bool:
        return self.cfg.features != "amplitude"

    def predict_amplitude(self, amplitude: np.ndarray) -> Prediction:
        return self._predict(preprocess_amplitude(amplitude, self.cfg))

    def predict_csi(self, csi: np.ndarray) -> Prediction:
        return self._predict(preprocess_csi(csi, self.cfg))

    @torch.no_grad()
    def _predict(self, features: np.ndarray) -> Prediction:
        x = torch.from_numpy(features).unsqueeze(0)
        probs = torch.softmax(self.model(x), dim=1)[0].numpy()
        best = int(probs.argmax())
        code = next(a for a, name in ACTIVITIES.items() if name == self.class_names[best])
        return Prediction(
            activity=self.class_names[best],
            activity_code=f"A{code}",
            confidence=float(probs[best]),
            probabilities={n: round(float(p), 6) for n, p in zip(self.class_names, probs, strict=True)},
        )

    def predict_csv_text(self, text: str) -> Prediction:
        _, csi = parse_csv_text(text)
        return self.predict_csi(csi)
