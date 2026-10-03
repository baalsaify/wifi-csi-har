"""REST API for activity prediction.

Run locally:
    uvicorn csi_har.api.main:app --reload
Then open http://127.0.0.1:8000/docs for the interactive OpenAPI page.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, model_validator

from csi_har import __version__
from csi_har.config import ACTIVITIES, N_CHANNELS
from csi_har.inference import HARPredictor

DEFAULT_MODEL_PATH = Path("models/har_cnn.pt")
MAX_UPLOAD_BYTES = 5 * 1024 * 1024


Matrix = list[list[float]]


def _check_matrix(name: str, rows: Matrix) -> None:
    if len(rows) < 32:
        raise ValueError(f"{name}: need at least 32 packets, got {len(rows)}")
    bad = next((i for i, r in enumerate(rows) if len(r) != N_CHANNELS), None)
    if bad is not None:
        raise ValueError(f"{name}: row {bad} has {len(rows[bad])} values, expected {N_CHANNELS}")


class CSIRequest(BaseModel):
    """Send either complex CSI (``csi_real`` + ``csi_imag``) or, for amplitude-only models, ``csi_amplitude``.

    Each matrix has one row per packet and 90 values per row (1 TX x 3 RX x 30 subcarriers),
    with at least 32 packets.
    """

    csi_real: Matrix | None = Field(None, description="Real part of the complex CSI")
    csi_imag: Matrix | None = Field(None, description="Imaginary part of the complex CSI")
    csi_amplitude: Matrix | None = Field(None, description="CSI amplitude (amplitude-only models)")

    @model_validator(mode="after")
    def check_inputs(self) -> CSIRequest:
        if self.csi_real is not None or self.csi_imag is not None:
            if self.csi_real is None or self.csi_imag is None:
                raise ValueError("send both csi_real and csi_imag")
            _check_matrix("csi_real", self.csi_real)
            _check_matrix("csi_imag", self.csi_imag)
            if len(self.csi_real) != len(self.csi_imag):
                raise ValueError("csi_real and csi_imag must have the same number of packets")
        elif self.csi_amplitude is not None:
            _check_matrix("csi_amplitude", self.csi_amplitude)
        else:
            raise ValueError("send csi_real + csi_imag, or csi_amplitude")
        return self

class PredictionResponse(BaseModel):
    activity: str
    activity_code: str
    confidence: float
    probabilities: dict[str, float]


def create_app(model_path: str | Path | None = None) -> FastAPI:
    path = Path(model_path or os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
    state: dict = {"predictor": None}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if path.exists():
            state["predictor"] = HARPredictor(path)
        yield

    app = FastAPI(title="Wi-Fi CSI Human Activity Recognition", version=__version__, lifespan=lifespan)

    def predictor() -> HARPredictor:
        if state["predictor"] is None:
            raise HTTPException(503, f"model not loaded (expected at {path})")
        return state["predictor"]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_loaded": state["predictor"] is not None, "version": __version__}

    @app.get("/model-info")
    def model_info() -> dict:
        p = predictor()
        return {
            "classes": {f"A{code}": name for code, name in ACTIVITIES.items()},
            "input": {"features": p.cfg.features, "csi_values_per_packet": N_CHANNELS,
                      "min_packets": p.cfg.min_packets, "resampled_length": p.cfg.target_length,
                      "accepts": ["csi_real + csi_imag"] + ([] if p.needs_phase else ["csi_amplitude"])},
            "training": p.extra,
        }

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: CSIRequest) -> PredictionResponse:
        p = predictor()
        try:
            if request.csi_real is not None:
                csi = np.asarray(request.csi_real) + 1j * np.asarray(request.csi_imag)
                result = p.predict_csi(csi)
            elif p.needs_phase:
                raise ValueError(f"this model uses {p.cfg.features!r} features; send csi_real and csi_imag")
            else:
                result = p.predict_amplitude(np.asarray(request.csi_amplitude, dtype=np.float32))
        except ValueError as err:
            raise HTTPException(422, str(err)) from err
        return PredictionResponse(**result.__dict__)

    @app.post("/predict/file", response_model=PredictionResponse)
    async def predict_file(file: UploadFile = File(..., description="One trial CSV from the dataset")):
        raw = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "file larger than 5 MB")
        try:
            result = predictor().predict_csv_text(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as err:
            raise HTTPException(422, f"could not read CSI file: {err}") from err
        return PredictionResponse(**result.__dict__)

    return app


app = create_app()
