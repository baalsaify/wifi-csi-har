"""Dataset constants and default hyperparameters."""

from dataclasses import dataclass, field
from pathlib import Path

# Activity codes as defined in Table 1 of Alsaify et al., Data in Brief 33 (2020) 106534.
# A7 and A9 are both "Turning" in the paper; they are the first and second turn of the
# walking experiment (C3), so they keep separate labels here.
ACTIVITIES: dict[int, str] = {
    1: "sit_still",
    2: "fall_from_sitting",
    3: "lie_down",
    4: "stand_still",
    5: "fall_from_standing",
    6: "walk_tx_to_rx",
    7: "turn_1",
    8: "walk_rx_to_tx",
    9: "turn_2",
    10: "stand_up",
    11: "sit_down",
    12: "pick_up_pen",
}
CLASS_NAMES: list[str] = [ACTIVITIES[a] for a in sorted(ACTIVITIES)]
N_CLASSES = len(CLASS_NAMES)

N_META_COLS = 13  # timestamp_low, bfee_count, Nrx, Ntx, rssi_a-c, noise, agc, perm_1-3, rate
N_SUBCARRIERS = 30
N_RX = 3
N_CHANNELS = N_SUBCARRIERS * N_RX  # 90 CSI streams (1 TX x 3 RX x 30 subcarriers)
SAMPLE_RATE_HZ = 320.0  # packets per second (paper, section 2)

MENDELEY_DATASET_ID = "v38wjmz6f6"
MENDELEY_VERSION = 1


N_PHASE_CHANNELS = (N_RX - 1) * N_SUBCARRIERS  # 60 phase differences (RX1-RX2, RX2-RX3)
FEATURE_SETS = {"amplitude": N_CHANNELS, "amplitude+phase": N_CHANNELS + N_PHASE_CHANNELS}


@dataclass(frozen=True)
class PreprocessConfig:
    features: str = "amplitude"  # one of FEATURE_SETS
    target_length: int = 256  # time steps after resampling
    hampel_half_window: int = 5
    hampel_n_sigmas: float = 3.0
    lowpass_cutoff_hz: float = 20.0
    lowpass_order: int = 4
    min_packets: int = 32


@dataclass
class TrainConfig:
    data_path: Path = Path("data/processed.npz")
    model_dir: Path = Path("models")
    report_dir: Path = Path("reports")
    n_folds: int = 5
    val_subjects: int = 3
    epochs: int = 40
    patience: int = 6
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    num_threads: int = 0  # 0 = let torch decide
    extra: dict = field(default_factory=dict)
