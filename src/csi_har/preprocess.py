"""Signal preprocessing for CSI amplitude streams.

Pipeline (per trial): |CSI| -> Hampel outlier filter -> Butterworth low-pass ->
linear resampling to a fixed length -> per-channel z-score. Output shape is
``(channels, target_length)`` so it feeds straight into a 1D-CNN.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt

from csi_har.config import N_CHANNELS, SAMPLE_RATE_HZ, PreprocessConfig

_MAD_SCALE = 1.4826  # makes the MAD a consistent estimator of the std for Gaussian noise


def hampel(x: np.ndarray, half_window: int = 5, n_sigmas: float = 3.0) -> np.ndarray:
    """Replace outliers along axis 0 with the rolling median. ``x`` is ``(time, channels)``."""
    size = (2 * half_window + 1, 1)
    med = median_filter(x, size=size, mode="nearest")
    deviation = np.abs(x - med)
    mad = _MAD_SCALE * median_filter(deviation, size=size, mode="nearest")
    outliers = deviation > n_sigmas * mad
    return np.where(outliers, med, x)


def lowpass(x: np.ndarray, cutoff_hz: float, fs_hz: float = SAMPLE_RATE_HZ, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass along axis 0."""
    b, a = butter(order, cutoff_hz / (fs_hz / 2), btype="low")
    return filtfilt(b, a, x, axis=0)


def resample_linear(x: np.ndarray, length: int) -> np.ndarray:
    """Linearly resample ``(time, channels)`` to ``(length, channels)``."""
    n = x.shape[0]
    src = np.arange(n)
    dst = np.linspace(0, n - 1, length)
    return np.stack([np.interp(dst, src, x[:, c]) for c in range(x.shape[1])], axis=1)


def zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-channel standardization along axis 0."""
    return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + eps)


def preprocess_amplitude(amplitude: np.ndarray, cfg: PreprocessConfig | None = None) -> np.ndarray:
    """``(n_packets, 90)`` amplitude -> ``(90, target_length)`` float32 model input."""
    cfg = cfg or PreprocessConfig()
    amplitude = np.asarray(amplitude, dtype=np.float64)
    if amplitude.ndim != 2 or amplitude.shape[1] != N_CHANNELS:
        raise ValueError(f"expected (n_packets, {N_CHANNELS}), got {amplitude.shape}")
    if amplitude.shape[0] < cfg.min_packets:
        raise ValueError(f"need at least {cfg.min_packets} packets, got {amplitude.shape[0]}")
    if not np.isfinite(amplitude).all():
        raise ValueError("amplitude contains NaN or inf")

    x = hampel(amplitude, cfg.hampel_half_window, cfg.hampel_n_sigmas)
    x = lowpass(x, cfg.lowpass_cutoff_hz, order=cfg.lowpass_order)
    x = resample_linear(x, cfg.target_length)
    x = zscore(x)
    return x.T.astype(np.float32)


def preprocess_csi(csi: np.ndarray, cfg: PreprocessConfig | None = None) -> np.ndarray:
    """Complex ``(n_packets, 90)`` CSI -> model input, using the amplitude only."""
    return preprocess_amplitude(np.abs(csi), cfg)
