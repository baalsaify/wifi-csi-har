import numpy as np
import pytest

from csi_har.config import PreprocessConfig
from csi_har.preprocess import (
    hampel,
    lowpass,
    phase_differences,
    preprocess_amplitude,
    preprocess_csi,
    resample_linear,
    zscore,
)


def _random_csi(n=400, seed=2):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 90)) + 1j * rng.normal(size=(n, 90))


def test_amplitude_plus_phase_features_shape():
    out = preprocess_csi(_random_csi(), PreprocessConfig(features="amplitude+phase", target_length=128))
    assert out.shape == (150, 128)
    assert np.isfinite(out).all()


def test_phase_difference_cancels_offset_shared_by_all_antennas():
    csi = _random_csi()
    offset = np.exp(1j * np.random.default_rng(3).uniform(-np.pi, np.pi, (len(csi), 1)))
    np.testing.assert_allclose(phase_differences(csi * offset), phase_differences(csi), atol=1e-9)


def test_amplitude_only_entry_point_rejects_phase_models():
    with pytest.raises(ValueError, match="complex CSI"):
        preprocess_amplitude(np.ones((100, 90)), PreprocessConfig(features="amplitude+phase"))


def test_hampel_removes_spike_and_keeps_clean_samples():
    x = np.ones((50, 2))
    x[25, 0] = 100.0
    y = hampel(x)
    assert y[25, 0] == pytest.approx(1.0)
    np.testing.assert_array_equal(y[:, 1], x[:, 1])


def test_lowpass_attenuates_high_frequency():
    t = np.arange(640) / 320.0
    slow = np.sin(2 * np.pi * 2 * t)
    fast = np.sin(2 * np.pi * 100 * t)
    y = lowpass((slow + fast)[:, None], cutoff_hz=20)
    assert np.abs(y[50:-50, 0] - slow[50:-50]).max() < 0.05


def test_resample_and_zscore_shapes():
    x = np.random.default_rng(0).normal(5, 3, (300, 4))
    r = resample_linear(x, 128)
    assert r.shape == (128, 4)
    z = zscore(r)
    np.testing.assert_allclose(z.mean(axis=0), 0, atol=1e-6)
    np.testing.assert_allclose(z.std(axis=0), 1, atol=1e-3)


def test_preprocess_amplitude_output():
    amp = np.abs(np.random.default_rng(1).normal(10, 2, (900, 90)))
    out = preprocess_amplitude(amp, PreprocessConfig(target_length=256))
    assert out.shape == (90, 256)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


@pytest.mark.parametrize("bad", [np.ones((10, 90)), np.ones((100, 89)), np.full((100, 90), np.nan)])
def test_preprocess_amplitude_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        preprocess_amplitude(bad)
