"""Read trial CSV files from the dataset (Intel 5300 CSI exported as text).

Each CSV row is one received packet: 13 numeric metadata columns followed by 90
complex CSI values written as ``a+bi`` (e.g. ``-11+13i`` or ``-20+-5i``).
"""

from __future__ import annotations

import re
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from csi_har.config import N_CHANNELS, N_META_COLS

TRIAL_NAME_RE = re.compile(r"E(\d+)_S(\d+)_C(\d+)_A(\d+)_T(\d+)\.csv$", re.IGNORECASE)
_VALUES_PER_ROW = N_META_COLS + 2 * N_CHANNELS


@dataclass(frozen=True)
class TrialInfo:
    environment: int
    subject: int
    experiment: int
    activity: int
    trial: int


def parse_trial_name(name: str) -> TrialInfo:
    """Parse ``E1_S04_C03_A07_T17.csv`` into its numeric fields."""
    match = TRIAL_NAME_RE.search(name)
    if not match:
        raise ValueError(f"not a dataset trial file name: {name!r}")
    return TrialInfo(*(int(g) for g in match.groups()))


def parse_csv_text(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse one trial file's text.

    Returns ``(meta, csi)`` where ``meta`` is float32 ``(n_packets, 13)`` and ``csi``
    is complex64 ``(n_packets, 90)``.
    """
    header, _, body = text.strip().partition("\n")
    columns = header.strip().split(",")
    if len(columns) != N_META_COLS + N_CHANNELS or not columns[N_META_COLS].startswith("csi_"):
        raise ValueError(f"unexpected header with {len(columns)} columns")
    if not body.strip():
        raise ValueError("file has a header but no packets")

    # "a+bi" -> "a b": turn every row into plain whitespace-separated numbers.
    flat = body.replace("i", "").replace("+", " ").replace(",", " ")
    values = np.array(flat.split(), dtype=np.float32)
    if values.size % _VALUES_PER_ROW:
        raise ValueError(f"{values.size} values is not a multiple of {_VALUES_PER_ROW} per row")

    rows = values.reshape(-1, _VALUES_PER_ROW)
    meta = rows[:, :N_META_COLS]
    csi = (rows[:, N_META_COLS::2] + 1j * rows[:, N_META_COLS + 1 :: 2]).astype(np.complex64)
    return meta, csi


def read_trial_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a single extracted trial CSV file."""
    return parse_csv_text(Path(path).read_text(encoding="utf-8"))


def iter_zip_trials(zip_path: str | Path) -> Iterator[tuple[TrialInfo, str]]:
    """Yield ``(TrialInfo, csv_text)`` for every trial inside a per-subject zip."""
    with zipfile.ZipFile(zip_path) as archive:
        for entry in sorted(archive.namelist()):
            if not entry.lower().endswith(".csv"):
                continue
            info = parse_trial_name(Path(entry).name)
            yield info, archive.read(entry).decode("utf-8")
