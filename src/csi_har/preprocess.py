"""Signal preprocessing for CSI streams.

Per stream: Hampel outlier filter -> Butterworth low-pass -> linear resampling to a fixed
length -> z-score. Two feature sets are supported:

- ``amplitude``: |CSI| for 3 RX x 30 subcarriers = 90 streams
- ``amplitude+phase``: the 90 amplitude streams plus 60 phase-difference streams between
  neighboring receive antennas (RX1-RX2, RX2-RX3). Antennas on one Intel 5300 NIC share an
  oscillator, so the difference cancels the random carrier/sampling phase offsets that make
  raw CSI phase unusable.

Output shape is ``(channels, target_length)`` so it feeds straight into a 1D-CNN.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt

from csi_har.config import FEATURE_SETS, N_CHANNELS, N_RX, N_SUBCARRIERS, SAMPLE_RATE_HZ, PreprocessConfig

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


def phase_differences(csi: np.ndarray) -> np.ndarray:
    """Complex ``(n_packets, 90)`` -> unwrapped phase differences ``(n_packets, 60)``."""
    c = csi.reshape(len(csi), N_RX, N_SUBCARRIERS)
    diff = np.angle(c[:, :-1] * np.conj(c[:, 1:]))  # (n, 2, 30): RX1-RX2, RX2-RX3
    return np.unwrap(diff, axis=0).reshape(len(csi), -1)


def _check(x: np.ndarray, width: int, cfg: PreprocessConfig, what: str) -> None:
    if x.ndim != 2 or x.shape[1] != width:
        raise ValueError(f"expected {what} of shape (n_packets, {width}), got {x.shape}")
    if x.shape[0] < cfg.min_packets:
        raise ValueError(f"need at least {cfg.min_packets} packets, got {x.shape[0]}")
    if not np.isfinite(x).all():
        raise ValueError(f"{what} contains NaN or inf")


def _clean(x: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    x = hampel(x, cfg.hampel_half_window, cfg.hampel_n_sigmas)
    x = lowpass(x, cfg.lowpass_cutoff_hz, order=cfg.lowpass_order)
    return zscore(resample_linear(x, cfg.target_length))


def preprocess_amplitude(amplitude: np.ndarray, cfg: PreprocessConfig | None = None) -> np.ndarray:
    """``(n_packets, 90)`` amplitude -> ``(90, target_length)`` float32 (amplitude-only models)."""
    cfg = cfg or PreprocessConfig()
    if cfg.features != "amplitude":
        raise ValueError(f"feature set {cfg.features!r} needs complex CSI, not amplitude only")
    amplitude = np.asarray(amplitude, dtype=np.float64)
    _check(amplitude, N_CHANNELS, cfg, "amplitude")
    return _clean(amplitude, cfg).T.astype(np.float32)


def preprocess_csi(csi: np.ndarray, cfg: PreprocessConfig | None = None) -> np.ndarray:
    """Complex ``(n_packets, 90)`` CSI -> ``(channels, target_length)`` float32 for any feature set."""
    cfg = cfg or PreprocessConfig()
    if cfg.features not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {cfg.features!r}; choose from {sorted(FEATURE_SETS)}")
    csi = np.asarray(csi, dtype=np.complex128)
    _check(np.abs(csi), N_CHANNELS, cfg, "CSI")
    parts = [_clean(np.abs(csi), cfg)]
    if cfg.features == "amplitude+phase":
        parts.append(_clean(phase_differences(csi), cfg))
    return np.concatenate(parts, axis=1).T.astype(np.float32)
