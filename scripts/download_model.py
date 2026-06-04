"""Download the TIDAL vR.P.30.1 inference model from Hugging Face Hub.

Usage:
    python scripts/download_model.py          # downloads to default location
    python scripts/download_model.py --force  # re-download even if already present
    HF_TOKEN=<your_token> python scripts/download_model.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration — mirrors model_loader.py / manifest.json
# ---------------------------------------------------------------------------
HF_REPO_ID = "the-harsh-vardhan/TIDAL"
HF_MODEL_FILE = "checkpoints/vR.P.30.1/best_model.pt"
HF_MANIFEST_FILE = "checkpoints/vR.P.30.1/manifest.json"

DEFAULT_MODEL_DIR = Path("models") / "inference" / "vR.P.30.1"
MODEL_FILENAME = "best_model.pt"
MANIFEST_FILENAME = "manifest.json"


def _sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def download_model(model_dir: Path, force: bool = False) -> None:
    """Download model checkpoint (and optional manifest) from HF Hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "[ERROR] huggingface_hub is not installed.\n"
            "        Run:  pip install huggingface_hub",
            file=sys.stderr,
        )
        sys.exit(1)

    model_dir.mkdir(parents=True, exist_ok=True)
    dest_model = model_dir / MODEL_FILENAME
    dest_manifest = model_dir / MANIFEST_FILENAME

    token = os.environ.get("HF_TOKEN")

    # ── checkpoint ────────────────────────────────────────────────────────
    if dest_model.exists() and not force:
        size_mb = dest_model.stat().st_size / 1e6
        print(f"[OK] Model already present at {dest_model}  ({size_mb:.1f} MB) — skipping.")
        print("     Use --force to re-download.")
    else:
        print(f"[->] Downloading checkpoint from {HF_REPO_ID} ...")
        try:
            downloaded = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_MODEL_FILE,
                local_dir=str(model_dir),
                local_dir_use_symlinks=False,
                token=token,
                force_download=force,
            )
            # hf_hub_download may place it inside a sub-folder; rename if needed
            downloaded_path = Path(downloaded)
            if downloaded_path != dest_model:
                downloaded_path.rename(dest_model)
            size_mb = dest_model.stat().st_size / 1e6
            sha = _sha256(dest_model)
            print(f"[OK] Saved → {dest_model}  ({size_mb:.1f} MB, sha256: {sha[:16]}...)")
        except Exception as exc:
            print(f"[ERROR] Download failed: {exc}", file=sys.stderr)
            sys.exit(1)

    # ── manifest (best-effort) ─────────────────────────────────────────────
    if not dest_manifest.exists() or force:
        try:
            dl = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_MANIFEST_FILE,
                local_dir=str(model_dir),
                local_dir_use_symlinks=False,
                token=token,
                force_download=force,
            )
            dl_path = Path(dl)
            if dl_path != dest_manifest:
                dl_path.rename(dest_manifest)
            print(f"[OK] Manifest saved → {dest_manifest}")
        except Exception:
            print("[WARN] manifest.json not found on Hub — that is fine if it exists locally.")

    # ── print summary ──────────────────────────────────────────────────────
    if dest_manifest.exists():
        try:
            info = json.loads(dest_manifest.read_text(encoding="utf-8"))
            print(f"\nModel: {info.get('bundle_name', '?')}")
            print(f"  version : {info.get('model_version', '?')}")
            print(f"  encoder : {info.get('model_config', {}).get('encoder_name', '?')}")
        except Exception:
            pass

    print("\n[DONE] Model is ready.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download TIDAL inference model from HF Hub.")
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help=f"Local destination directory (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file already exists.",
    )
    args = parser.parse_args()

    download_model(Path(args.model_dir), force=args.force)


if __name__ == "__main__":
    main()
