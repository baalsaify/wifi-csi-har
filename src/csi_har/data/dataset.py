"""Build a preprocessed, cached array dataset from the downloaded per-subject zips.

Usage:
    python -m csi_har.data.dataset --raw data/raw --out data/processed.npz
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from csi_har.config import PreprocessConfig
from csi_har.data.reader import iter_zip_trials, parse_csv_text
from csi_har.preprocess import preprocess_csi

FIELDS = ("environment", "subject", "experiment", "activity", "trial")


def _process_zip(zip_path: str, cfg: PreprocessConfig) -> tuple[np.ndarray, np.ndarray, list[str]]:
    xs, infos, skipped = [], [], []
    for info, text in iter_zip_trials(zip_path):
        try:
            _, csi = parse_csv_text(text)
            xs.append(preprocess_csi(csi, cfg))
            infos.append([getattr(info, f) for f in FIELDS])
        except ValueError as err:
            skipped.append(f"{Path(zip_path).name}: {info} ({err})")
    return np.stack(xs), np.asarray(infos, dtype=np.int16), skipped


def build_cache(raw_dir: Path, out_path: Path, workers: int | None = None,
                cfg: PreprocessConfig | None = None) -> dict:
    cfg = cfg or PreprocessConfig()
    zips = sorted(raw_dir.rglob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"no zip files under {raw_dir}; run csi_har.data.download first")

    start = time.time()
    xs, infos, skipped = [], [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (x, info, skip) in enumerate(pool.map(_process_zip, map(str, zips), [cfg] * len(zips)), 1):
            xs.append(x)
            infos.append(info)
            skipped += skip
            print(f"[{i}/{len(zips)}] {x.shape[0]} trials ({time.time() - start:.0f}s)")

    meta = np.concatenate(infos)
    arrays = {"X": np.concatenate(xs).astype(np.float32)}
    arrays.update({name: meta[:, k] for k, name in enumerate(FIELDS)})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, target_length=cfg.target_length, **arrays)

    summary = {"trials": int(arrays["X"].shape[0]), "skipped": skipped,
               "subjects": int(len(np.unique(arrays["subject"]))), "seconds": round(time.time() - start)}
    print(summary)
    return summary


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed.npz"))
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    build_cache(args.raw, args.out, args.workers)


if __name__ == "__main__":
    main()
