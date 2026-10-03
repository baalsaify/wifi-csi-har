"""Download the dataset from Mendeley Data (public API, CC BY 4.0).

Usage:
    python -m csi_har.data.download --out data/raw            # all 3 environments (~2.1 GB)
    python -m csi_har.data.download --out data/raw --env 1    # one environment (~0.7 GB)

Files are verified against the SHA-256 published by Mendeley and skipped if already present.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import httpx

from csi_har.config import MENDELEY_DATASET_ID, MENDELEY_VERSION

API = f"https://data.mendeley.com/public-api/datasets/{MENDELEY_DATASET_ID}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_files(client: httpx.Client) -> list[dict]:
    """Return ``[{folder, filename, size, sha256, url}]`` for every file in the dataset."""
    folders = client.get(f"{API}/folders/{MENDELEY_VERSION}").raise_for_status().json()
    files = []
    for folder in folders:
        listing = client.get(
            f"{API}/files", params={"folder_id": folder["id"], "version": MENDELEY_VERSION}
        ).raise_for_status().json()
        for item in listing:
            details = item["content_details"]
            files.append(
                {
                    "folder": folder["name"],
                    "filename": item["filename"],
                    "size": details["size"],
                    "sha256": details["sha256_hash"],
                    "url": details["download_url"],
                }
            )
    return files


def download(out_dir: Path, environments: set[int] | None = None) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        for f in list_files(client):
            env = int(f["folder"].split()[-1])
            if environments and env not in environments:
                continue
            target = out_dir / f["folder"] / f["filename"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size == f["size"] and _sha256(target) == f["sha256"]:
                print(f"ok    {target}")
                paths.append(target)
                continue
            print(f"fetch {target} ({f['size'] / 1e6:.0f} MB)")
            tmp = target.with_suffix(target.suffix + ".part")
            with client.stream("GET", f["url"]) as response, tmp.open("wb") as fh:
                response.raise_for_status()
                for chunk in response.iter_bytes(1 << 20):
                    fh.write(chunk)
            if _sha256(tmp) != f["sha256"]:
                tmp.unlink()
                raise RuntimeError(f"checksum mismatch for {target.name}")
            tmp.replace(target)
            paths.append(target)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument("--env", type=int, nargs="*", choices=[1, 2, 3], help="environments to fetch")
    args = parser.parse_args()
    paths = download(args.out, set(args.env) if args.env else None)
    print(f"{len(paths)} files ready in {args.out}")


if __name__ == "__main__":
    main()
