from pathlib import Path

import numpy as np
import pytest

from csi_har.config import N_CHANNELS, N_SUBCARRIERS, PreprocessConfig
from csi_har.inference import save_model
from csi_har.models import HARCNN

META_HEADER = ["timestamp_low", "bfee_count", "Nrx", "Ntx", "rssi_a", "rssi_b", "rssi_c",
               "noise", "agc", "perm_1", "perm_2", "perm_3", "rate"]


def make_csv_text(n_packets: int = 64, seed: int = 0) -> tuple[str, np.ndarray]:
    """Build a dataset-format CSV string and return it with the complex CSI it encodes."""
    rng = np.random.default_rng(seed)
    csi = rng.integers(-30, 30, (n_packets, N_CHANNELS)) + 1j * rng.integers(-30, 30, (n_packets, N_CHANNELS))
    header = META_HEADER + [f"csi_1_{rx}_{sc}" for rx in range(1, 4) for sc in range(1, N_SUBCARRIERS + 1)]
    lines = [",".join(header)]
    for i, row in enumerate(csi):
        meta = [80068132 + 3125 * i, 25489 + i, 3, 1, 40, 40, 34, -127, 44, 2, 1, 3, 256]
        values = [f"{int(c.real)}+{int(c.imag)}i" for c in row]  # e.g. "-20+-5i", as in the dataset
        lines.append(",".join(map(str, meta)) + "," + ",".join(values))
    return "\n".join(lines) + "\n", csi


@pytest.fixture
def csv_text() -> tuple[str, np.ndarray]:
    return make_csv_text()


@pytest.fixture
def tiny_model_path(tmp_path: Path) -> Path:
    """An untrained model checkpoint, so API tests need neither data nor training."""
    path = tmp_path / "model.pt"
    save_model(HARCNN(width=8), path, PreprocessConfig(target_length=64), extra={"note": "test"})
    return path
