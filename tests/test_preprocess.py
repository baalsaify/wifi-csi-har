import numpy as np
import pytest

from csi_har.config import PreprocessConfig
from csi_har.preprocess import hampel, lowpass, preprocess_amplitude, resample_linear, zscore


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
