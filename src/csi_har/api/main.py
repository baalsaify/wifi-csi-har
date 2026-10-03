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
from pydantic import BaseModel, Field, field_validator

from csi_har import __version__
from csi_har.config import ACTIVITIES, N_CHANNELS
from csi_har.inference import HARPredictor

DEFAULT_MODEL_PATH = Path("models/har_cnn.pt")
MAX_UPLOAD_BYTES = 5 * 1024 * 1024


class AmplitudeRequest(BaseModel):
    csi_amplitude: list[list[float]] = Field(
        ..., description=f"CSI amplitude matrix, one row per packet, {N_CHANNELS} values per row "
                         "(1 TX x 3 RX x 30 subcarriers). At least 32 packets."
    )

    @field_validator("csi_amplitude")
    @classmethod
    def check_shape(cls, rows: list[list[float]]) -> list[list[float]]:
        if len(rows) < 32:
            raise ValueError(f"need at least 32 packets, got {len(rows)}")
        bad = next((i for i, r in enumerate(rows) if len(r) != N_CHANNELS), None)
        if bad is not None:
            raise ValueError(f"row {bad} has {len(rows[bad])} values, expected {N_CHANNELS}")
        return rows


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
            "input": {"channels": N_CHANNELS, "min_packets": p.cfg.min_packets,
                      "resampled_length": p.cfg.target_length},
            "training": p.extra,
        }

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: AmplitudeRequest) -> PredictionResponse:
        try:
            result = predictor().predict_amplitude(np.asarray(request.csi_amplitude, dtype=np.float32))
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
