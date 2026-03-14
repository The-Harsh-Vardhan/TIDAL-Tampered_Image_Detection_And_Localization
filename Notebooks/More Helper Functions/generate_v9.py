from __future__ import annotations

import ast
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "Notebooks"


def _source(text: str) -> list[str]:
    text = dedent(text).strip("\n")
    if not text:
        return []
    return [line + "\n" for line in text.splitlines()]


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _source(text),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def validate_code_cells(cells: list[dict]) -> None:
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        try:
            ast.parse(source)
        except SyntaxError as exc:
            raise SyntaxError(
                f"Code cell {index} failed syntax validation: {exc.msg} at line {exc.lineno}"
            ) from exc


def install_cell() -> dict:
    return code(
        """
        import importlib
        import subprocess
        import sys

        REQUIRED_PACKAGES = {
            "albumentations": "albumentations>=1.4.0",
            "cv2": "opencv-python-headless>=4.9.0.80",
            "segmentation_models_pytorch": "segmentation-models-pytorch>=0.3.3",
            "imagehash": "ImageHash>=4.3.1",
            "skimage": "scikit-image>=0.22.0",
            "scipy": "scipy>=1.11.0",
            "wandb": "wandb>=0.17.0",
            "kaggle": "kaggle>=1.6.17",
        }

        for module_name, package_name in REQUIRED_PACKAGES.items():
            try:
                importlib.import_module(module_name)
            except ImportError:
                print(f"Installing {package_name} ...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package_name])

        print("Dependency check complete.")
        """
    )


def imports_cell() -> dict:
    return code(
        """
        import copy
        import gc
        import io
        import json
        import math
        import os
        import random
        import re
        import shutil
        import time
        import warnings
        from collections import Counter, defaultdict
        from contextlib import nullcontext
        from pathlib import Path

        import albumentations as A
        import cv2
        import imagehash
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        import segmentation_models_pytorch as smp
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import wandb
        from PIL import Image
        from IPython.display import Markdown, display
        from scipy.ndimage import binary_dilation
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            f1_score,
            precision_recall_curve,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.model_selection import train_test_split
        from skimage.segmentation import find_boundaries
        from torch.cuda.amp import GradScaler
        from torch.utils.data import DataLoader, Dataset
        from tqdm.auto import tqdm

        sns.set_style("whitegrid")
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        plt.rcParams["figure.dpi"] = 120
        """
    )


def config_cell() -> dict:
    return code(
        """
        CONFIG = {
            "experiment_name": "v9-casia-dualtask",
            "seed": 42,
            "split_seed": 42,
            "notes": "Audit9-grounded v9 notebook built from v8 run-01 lessons and only approved Docs9 changes.",
            "dataset_name": "CASIA v2.0",
            "data_dir": None,
            "output_dir": None,
            "kaggle_dataset_slug": "",
            "colab_drive_root": "/content/drive/MyDrive/BigVision Assignment",
            "colab_data_root": "/content/data",
            "colab_download_if_missing": False,
            "split_train": 0.70,
            "split_val": 0.15,
            "split_test": 0.15,
            "phash_hash_size": 8,
            "phash_distance_threshold": 4,
            "phash_bands": 4,
            "image_size": 384,
            "batch_size": 4,
            "num_workers": 2,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 2,
            "use_ela": True,
            "ela_quality": 90,
            "architecture": "unet",
            "encoder_name": "resnet34",
            "pretrained": True,
            "use_dual_task": True,
            "cls_loss_weight": 0.5,
            "use_edge_loss": True,
            "edge_loss_weight": 2.0,
            "edge_loss_lambda": 0.3,
            "decoder_dropout": 0.3,
            "max_epochs": 50,
            "encoder_lr": 1e-4,
            "decoder_lr": 1e-3,
            "weight_decay": 1e-4,
            "accumulation_steps": 4,
            "max_grad_norm": 1.0,
            "use_amp": True,
            "early_stopping_patience": 10,
            "save_every_n_epochs": 5,
            "scheduler_patience": 3,
            "scheduler_factor": 0.5,
            "min_lr": 1e-6,
            "aug_hflip": True,
            "aug_vflip": True,
            "aug_rotate90": True,
            "aug_color_jitter": True,
            "aug_compression": True,
            "aug_gauss_noise": True,
            "aug_gauss_blur": True,
            "threshold_min": 0.05,
            "threshold_max": 0.80,
            "threshold_step": 0.05,
            "boundary_tolerance": 2,
            "max_pixel_pr_samples": 1000000,
            "visualization_examples": 6,
            "mask_randomization_iterations": 5,
            "track_forgery_type_loss": True,
            "run_mask_randomization": True,
            "run_robustness_suite": True,
            "run_primary_training": True,
            "run_multi_seed_validation": False,
            "multi_seed_values": [42, 123, 789],
            "run_architecture_comparison": False,
            "run_augmentation_ablation": False,
            "use_wandb": False,
            "wandb_project": "tampered-image-detection",
            "wandb_entity": None,
            "wandb_mode": "disabled",
        }

        CONFIG["in_channels"] = 4 if CONFIG["use_ela"] else 3
        CONFIG["primary_selection_metric"] = "best_pixel_f1"
        CONFIG["robustness_degradations"] = [
            "clean",
            "jpeg_qf70",
            "jpeg_qf50",
            "gaussian_noise_light",
            "gaussian_noise_heavy",
            "gaussian_blur",
            "resize_0.75x",
            "resize_0.5x",
        ]
        CONFIG["thresholds"] = np.arange(
            CONFIG["threshold_min"],
            CONFIG["threshold_max"] + 1e-9,
            CONFIG["threshold_step"],
        ).round(4).tolist()

        print(json.dumps(CONFIG, indent=2))
        """
    )


def runtime_cell(target: str) -> dict:
    if target == "kaggle":
        return code(
            """
            ENVIRONMENT = "kaggle"
            INPUT_ROOT = Path("/kaggle/input")
            WORK_ROOT = Path("/kaggle/working")
            WORK_ROOT.mkdir(parents=True, exist_ok=True)

            if CONFIG["data_dir"]:
                data_roots = [Path(CONFIG["data_dir"])]
            else:
                data_roots = [INPUT_ROOT]

            CONFIG["output_dir"] = str(WORK_ROOT / CONFIG["experiment_name"])
            Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)

            print(f"Environment: {ENVIRONMENT}")
            print(f"Input roots: {[str(p) for p in data_roots]}")
            print(f"Artifacts: {CONFIG['output_dir']}")
            """
        )

    return code(
        """
        ENVIRONMENT = "colab"
        _ipython = globals().get("get_ipython", lambda: None)()
        IN_COLAB = _ipython is not None and "google.colab" in str(type(_ipython)).lower()

        if IN_COLAB:
            from google.colab import drive

            drive.mount("/content/drive", force_remount=False)
        else:
            print("Not running inside Google Colab. The notebook still defines Colab-compatible paths.")

        drive_root = Path(CONFIG["colab_drive_root"])
        local_data_root = Path(CONFIG["colab_data_root"])
        local_data_root.mkdir(parents=True, exist_ok=True)

        data_roots = []
        if CONFIG["data_dir"]:
            data_roots.append(Path(CONFIG["data_dir"]))
        data_roots.extend([drive_root, local_data_root, Path("/content")])

        artifact_root = drive_root / "artifacts" if IN_COLAB else local_data_root / "artifacts"
        CONFIG["output_dir"] = str((artifact_root / CONFIG["experiment_name"]).resolve())
        Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)

        if CONFIG["colab_download_if_missing"] and CONFIG["kaggle_dataset_slug"]:
            candidate_metadata = []
            for search_root in data_roots:
                if search_root.exists():
                    candidate_metadata.extend(search_root.rglob("metadata.csv"))

            if not candidate_metadata:
                kaggle_credentials = [
                    Path("/content/kaggle.json"),
                    Path("/content/drive/MyDrive/kaggle/kaggle.json"),
                    Path.home() / ".kaggle" / "kaggle.json",
                ]
                kaggle_json = next((path for path in kaggle_credentials if path.exists()), None)
                if kaggle_json is None:
                    print("kaggle.json not found. Skipping automatic dataset download.")
                else:
                    kaggle_dir = Path.home() / ".kaggle"
                    kaggle_dir.mkdir(parents=True, exist_ok=True)
                    target_json = kaggle_dir / "kaggle.json"
                    if kaggle_json != target_json:
                        shutil.copy2(kaggle_json, target_json)
                    target_json.chmod(0o600)

                    from kaggle.api.kaggle_api_extended import KaggleApi

                    api = KaggleApi()
                    api.authenticate()
                    download_root = local_data_root / "kaggle_download"
                    download_root.mkdir(parents=True, exist_ok=True)
                    print(f"Downloading dataset {CONFIG['kaggle_dataset_slug']} to {download_root}")
                    api.dataset_download_files(
                        CONFIG["kaggle_dataset_slug"],
                        path=str(download_root),
                        unzip=True,
                    )
                    data_roots.insert(0, download_root)

        print(f"Environment: {ENVIRONMENT}")
        print(f"Input roots: {[str(p) for p in data_roots]}")
        print(f"Artifacts: {CONFIG['output_dir']}")
        """
    )


def setup_cell() -> dict:
    return code(
        """
        def set_seed(seed: int) -> None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


        def seed_worker(worker_id: int) -> None:
            worker_seed = CONFIG["seed"] + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)


        def start_wandb_run(config: dict, run_name: str):
            if not config.get("use_wandb", False):
                return None
            try:
                run = wandb.init(
                    project=config["wandb_project"],
                    entity=config.get("wandb_entity"),
                    mode=config.get("wandb_mode", "online"),
                    config=config,
                    name=run_name,
                    reinit=True,
                )
                return run
            except Exception as exc:
                print(f"W&B disabled because initialization failed: {exc}")
                return None


        def finish_wandb_run(run) -> None:
            if run is not None:
                run.finish()


        def ensure_dir(pathlike) -> Path:
            path = Path(pathlike)
            path.mkdir(parents=True, exist_ok=True)
            return path


        def to_serializable(value):
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, (np.integer,)):
                return int(value)
            if isinstance(value, (np.floating,)):
                return float(value)
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().tolist()
            if isinstance(value, dict):
                return {str(k): to_serializable(v) for k, v in value.items()}
            if isinstance(value, list):
                return [to_serializable(v) for v in value]
            return value


        def save_json(pathlike, payload: dict) -> None:
            path = ensure_dir(Path(pathlike).parent) / Path(pathlike).name
            with open(path, "w", encoding="utf-8") as f:
                json.dump(to_serializable(payload), f, indent=2)


        set_seed(CONFIG["seed"])
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        AMP_ENABLED = bool(CONFIG["use_amp"] and DEVICE.type == "cuda")
        OUTPUT_ROOT = ensure_dir(CONFIG["output_dir"])

        print(f"Device: {DEVICE}")
        print(f"AMP enabled: {AMP_ENABLED}")
        print(f"Output root: {OUTPUT_ROOT}")
        """
    )


def dataset_discovery_cell() -> dict:
    return code(
        """
        IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        MASK_HINTS = ("mask", "groundtruth", "ground_truth", "gt", "binary", "forged")
        AUTH_HINTS = ("auth", "authentic", "original", "clean", "real", "au")
        TAMPER_HINTS = ("tamper", "tampered", "forged", "splice", "splicing", "tp", "copy_move", "copy-move", "copymove")


        def normalize_key(path_or_name: str) -> str:
            stem = Path(path_or_name).stem.lower()
            stem = re.sub(r"(?:_mask|_gt|_groundtruth|_forged|_tampered|_binary)+$", "", stem)
            stem = re.sub(r"[^a-z0-9]+", "", stem)
            return stem


        def contains_any(text: str, hints) -> bool:
            text = text.lower()
            return any(hint in text for hint in hints)


        def path_tokens(pathlike) -> list[str]:
            return [token for token in re.split(r"[^a-z0-9]+", str(pathlike).lower()) if token]


        def is_mask_file(path: Path) -> bool:
            joined = " ".join(path_tokens(path))
            return contains_any(joined, MASK_HINTS)


        def load_pairs_from_metadata(metadata_path: Path) -> list[dict]:
            df = pd.read_csv(metadata_path)
            required = {"image_path", "mask_path"}
            if not required.issubset(df.columns):
                raise ValueError(f"{metadata_path} must include columns {required}")

            pairs = []
            for _, row in df.iterrows():
                image_path = (metadata_path.parent / str(row["image_path"])).resolve()
                mask_path_value = row["mask_path"]
                mask_path = None
                if pd.notna(mask_path_value) and str(mask_path_value).strip():
                    mask_path = (metadata_path.parent / str(mask_path_value)).resolve()
                label = 1 if mask_path else 0
                forgery_type = str(row.get("forgery_type", "")).strip().lower()
                if not forgery_type:
                    combined_text = f"{image_path} {mask_path or ''}".lower()
                    forgery_type = "copymove" if "copy" in combined_text else ("authentic" if label == 0 else "splicing")

                pairs.append(
                    {
                        "image_path": str(image_path),
                        "mask_path": str(mask_path) if mask_path else None,
                        "label": label,
                        "forgery_type": forgery_type,
                        "source": str(row.get("source", "casia")),
                        "sample_id": normalize_key(image_path.name),
                    }
                )
            return pairs


        def discover_dataset_root(search_roots: list[Path]) -> Path:
            metadata_candidates = []
            for root in search_roots:
                if root is None:
                    continue
                root = Path(root)
                if not root.exists():
                    continue
                if (root / "metadata.csv").exists():
                    metadata_candidates.append(root)
                metadata_candidates.extend(path.parent for path in root.rglob("metadata.csv"))

            if metadata_candidates:
                ranked = sorted(set(metadata_candidates), key=lambda path: (len(path.parts), str(path)))
                return ranked[0]

            for root in search_roots:
                if root is None:
                    continue
                root = Path(root)
                if root.exists() and any(root.rglob("*")):
                    return root

            raise FileNotFoundError("Could not discover a dataset root. Set CONFIG['data_dir'] explicitly.")


        def infer_forgery_type(path_text: str, label: int) -> str:
            text = str(path_text).lower()
            if "copy" in text:
                return "copymove"
            if "splice" in text or "splicing" in text:
                return "splicing"
            return "authentic" if label == 0 else "splicing"


        def discover_pairs_without_metadata(data_root: Path) -> tuple[list[dict], dict]:
            image_files = []
            mask_files = []
            for path in data_root.rglob("*"):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                if is_mask_file(path):
                    mask_files.append(path)
                else:
                    image_files.append(path)

            if not image_files:
                raise RuntimeError(f"No images found under {data_root}")

            mask_lookup = defaultdict(list)
            for mask_path in mask_files:
                mask_lookup[normalize_key(mask_path.name)].append(mask_path)

            pairs = []
            skipped_missing_masks = []
            for image_path in sorted(image_files):
                image_text = str(image_path).lower()
                key = normalize_key(image_path.name)
                mask_candidates = mask_lookup.get(key, [])
                mask_path = mask_candidates[0] if mask_candidates else None

                label_hint = contains_any(image_text, TAMPER_HINTS)
                auth_hint = contains_any(image_text, AUTH_HINTS)
                label = 1 if mask_path is not None or (label_hint and not auth_hint) else 0

                if label == 1 and mask_path is None:
                    skipped_missing_masks.append(str(image_path))
                    continue

                pairs.append(
                    {
                        "image_path": str(image_path.resolve()),
                        "mask_path": str(mask_path.resolve()) if mask_path else None,
                        "label": label,
                        "forgery_type": infer_forgery_type(f"{image_path} {mask_path or ''}", label),
                        "source": "casia",
                        "sample_id": normalize_key(image_path.name),
                    }
                )

            summary = {
                "image_count": len(image_files),
                "mask_count": len(mask_files),
                "pairs_count": len(pairs),
                "skipped_missing_masks": len(skipped_missing_masks),
                "skipped_examples": skipped_missing_masks[:10],
            }
            return pairs, summary


        def summarize_pairs(pairs: list[dict]) -> pd.DataFrame:
            df = pd.DataFrame(pairs)
            if df.empty:
                raise RuntimeError("Pair discovery returned zero samples.")
            df = df.copy()
            df["label"] = df["label"].astype(int)
            type_counts = df["forgery_type"].value_counts().rename_axis("forgery_type").reset_index(name="count")
            display(type_counts)
            summary = pd.DataFrame(
                {
                    "num_images": [len(df)],
                    "num_tampered": [int(df["label"].sum())],
                    "num_authentic": [int((1 - df["label"]).sum())],
                    "num_masked": [int(df["mask_path"].notna().sum())],
                }
            )
            display(summary)
            return df


        DATASET_ROOT = discover_dataset_root(data_roots)
        metadata_csv = next(iter(sorted(DATASET_ROOT.rglob("metadata.csv"))), None)

        if metadata_csv is not None:
            PAIRS = load_pairs_from_metadata(metadata_csv)
            DISCOVERY_REPORT = {
                "dataset_root": str(DATASET_ROOT),
                "metadata_csv": str(metadata_csv),
                "from_metadata": True,
            }
        else:
            PAIRS, DISCOVERY_REPORT = discover_pairs_without_metadata(DATASET_ROOT)
            DISCOVERY_REPORT.update(
                {
                    "dataset_root": str(DATASET_ROOT),
                    "metadata_csv": None,
                    "from_metadata": False,
                }
            )

        PAIRS_DF = summarize_pairs(PAIRS)
        PAIRS_DF = PAIRS_DF.reset_index(drop=True)

        print("Discovery report:")
        print(json.dumps(DISCOVERY_REPORT, indent=2))
        """
    )


def phash_cell() -> dict:
    return code(
        """
        def phash_to_int(hash_value) -> int:
            return int(str(hash_value), 16)


        def hamming_distance(a: int, b: int) -> int:
            return (a ^ b).bit_count()


        class UnionFind:
            def __init__(self, n: int):
                self.parent = list(range(n))

            def find(self, x: int) -> int:
                while self.parent[x] != x:
                    self.parent[x] = self.parent[self.parent[x]]
                    x = self.parent[x]
                return x

            def union(self, a: int, b: int) -> None:
                ra, rb = self.find(a), self.find(b)
                if ra != rb:
                    self.parent[rb] = ra


        def build_duplicate_groups(pair_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
            records = pair_df.reset_index(drop=True).to_dict("records")
            hash_records = []
            for row in tqdm(records, desc="Computing pHash", leave=False):
                with Image.open(row["image_path"]) as img:
                    ph = imagehash.phash(img.convert("RGB"), hash_size=CONFIG["phash_hash_size"])
                hex_hash = str(ph)
                hash_records.append(
                    {
                        "image_path": row["image_path"],
                        "sample_id": row["sample_id"],
                        "label": int(row["label"]),
                        "forgery_type": str(row["forgery_type"]),
                        "hash_hex": hex_hash,
                        "hash_int": phash_to_int(ph),
                    }
                )

            if not hash_records:
                raise RuntimeError("No images available for pHash grouping.")

            bands = CONFIG["phash_bands"]
            hash_length = len(hash_records[0]["hash_hex"])
            segment_length = max(1, hash_length // max(bands, 1))
            buckets = defaultdict(list)
            for idx, item in enumerate(hash_records):
                for band_idx in range(bands):
                    start = band_idx * segment_length
                    end = hash_length if band_idx == bands - 1 else min(hash_length, start + segment_length)
                    buckets[(band_idx, item["hash_hex"][start:end])].append(idx)

            candidate_pairs = set()
            for bucket in buckets.values():
                if len(bucket) < 2:
                    continue
                for left_index in range(len(bucket)):
                    for right_index in range(left_index + 1, len(bucket)):
                        pair = tuple(sorted((bucket[left_index], bucket[right_index])))
                        candidate_pairs.add(pair)

            uf = UnionFind(len(hash_records))
            near_duplicate_examples = []
            for idx_a, idx_b in tqdm(sorted(candidate_pairs), desc="Checking near-duplicates", leave=False):
                item_a = hash_records[idx_a]
                item_b = hash_records[idx_b]
                distance = hamming_distance(item_a["hash_int"], item_b["hash_int"])
                if distance <= CONFIG["phash_distance_threshold"]:
                    uf.union(idx_a, idx_b)
                    if len(near_duplicate_examples) < 10:
                        near_duplicate_examples.append(
                            {
                                "distance": int(distance),
                                "left": item_a,
                                "right": item_b,
                            }
                        )

            groups = defaultdict(list)
            for idx, row in enumerate(hash_records):
                groups[uf.find(idx)].append(row)

            updated_df = pair_df.copy()
            group_rows = []
            duplicate_group_count = 0
            max_group_size = 1
            for group_index, members in enumerate(groups.values()):
                duplicate_group_id = f"group_{group_index:05d}"
                paths = {item["image_path"] for item in members}
                updated_df.loc[updated_df["image_path"].isin(paths), "duplicate_group_id"] = duplicate_group_id
                labels = [int(item["label"]) for item in members]
                tampered_types = [item["forgery_type"] for item in members if int(item["label"]) == 1]
                group_label = 1 if any(labels) else 0
                group_type = Counter(tampered_types).most_common(1)[0][0] if tampered_types else "authentic"
                group_rows.append(
                    {
                        "duplicate_group_id": duplicate_group_id,
                        "label": group_label,
                        "forgery_type": group_type,
                        "group_size": len(members),
                        "stratify_key": f"{group_label}_{group_type}",
                    }
                )
                if len(members) > 1:
                    duplicate_group_count += 1
                max_group_size = max(max_group_size, len(members))

            group_df = pd.DataFrame(group_rows)
            report = {
                "hash_size": CONFIG["phash_hash_size"],
                "distance_threshold": CONFIG["phash_distance_threshold"],
                "num_hashes": len(hash_records),
                "candidate_pairs": len(candidate_pairs),
                "num_groups": int(len(group_df)),
                "near_duplicate_group_count": int(duplicate_group_count),
                "max_group_size": int(max_group_size),
                "examples": near_duplicate_examples,
            }
            return updated_df, group_df, report


        def safe_group_train_test(group_df: pd.DataFrame, test_size: float, seed: int):
            stratify = group_df["stratify_key"]
            counts = stratify.value_counts()
            if counts.empty or counts.min() < 2:
                stratify = None
            return train_test_split(group_df, test_size=test_size, random_state=seed, stratify=stratify)


        def assign_group_splits(pair_df: pd.DataFrame, group_df: pd.DataFrame, seed: int) -> pd.DataFrame:
            train_groups, temp_groups = safe_group_train_test(group_df, 1.0 - CONFIG["split_train"], seed)
            relative_test = CONFIG["split_test"] / (CONFIG["split_val"] + CONFIG["split_test"])
            val_groups, test_groups = safe_group_train_test(temp_groups, relative_test, seed)

            split_lookup = {}
            for split_name, split_groups in [("train", train_groups), ("val", val_groups), ("test", test_groups)]:
                for group_id in split_groups["duplicate_group_id"].tolist():
                    split_lookup[group_id] = split_name

            split_df = pair_df.copy()
            split_df["split"] = split_df["duplicate_group_id"].map(split_lookup)
            if split_df["split"].isna().any():
                missing = split_df[split_df["split"].isna()][["image_path", "duplicate_group_id"]]
                raise RuntimeError(f"Some samples were not assigned to a split: {missing.head().to_dict('records')}")
            return split_df


        PAIRS_DF, GROUP_DF, PHASH_REPORT = build_duplicate_groups(PAIRS_DF)
        SPLIT_DF = assign_group_splits(PAIRS_DF, GROUP_DF, CONFIG["split_seed"])

        for split_name in ["train", "val", "test"]:
            split_count = int((SPLIT_DF["split"] == split_name).sum())
            print(f"{split_name}: {split_count} samples")

        split_table = (
            SPLIT_DF.groupby(["split", "forgery_type", "label"])
            .size()
            .reset_index(name="count")
            .sort_values(["split", "forgery_type", "label"])
        )
        display(split_table)

        print("pHash report:")
        print(json.dumps(PHASH_REPORT, indent=2))

        SPLITS = {
            split_name: SPLIT_DF[SPLIT_DF["split"] == split_name].to_dict("records")
            for split_name in ["train", "val", "test"]
        }

        split_manifest_path = OUTPUT_ROOT / "split_manifest.csv"
        SPLIT_DF.to_csv(split_manifest_path, index=False)
        save_json(
            OUTPUT_ROOT / "data_discovery_report.json",
            {
                "discovery": DISCOVERY_REPORT,
                "phash": PHASH_REPORT,
                "group_count": int(len(GROUP_DF)),
                "split_seed": int(CONFIG["split_seed"]),
            },
        )
        print(f"Saved split manifest to {split_manifest_path}")
        """
    )


def dataset_pipeline_cell() -> dict:
    return code(
        """
        IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


        def compute_ela(image_bgr: np.ndarray, quality: int = 90) -> np.ndarray:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
            success, encoded = cv2.imencode(".jpg", image_bgr, encode_param)
            if not success:
                return np.zeros(image_bgr.shape[:2], dtype=np.uint8)
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            ela = cv2.absdiff(image_bgr, decoded)
            ela_gray = cv2.cvtColor(ela, cv2.COLOR_BGR2GRAY)
            return ela_gray


        def build_spatial_transform(training: bool, config: dict):
            transforms = [A.Resize(config["image_size"], config["image_size"])]
            if training and config["aug_hflip"]:
                transforms.append(A.HorizontalFlip(p=0.5))
            if training and config["aug_vflip"]:
                transforms.append(A.VerticalFlip(p=0.5))
            if training and config["aug_rotate90"]:
                transforms.append(A.RandomRotate90(p=0.5))
            return A.Compose(transforms)


        def build_rgb_only_transform(training: bool, config: dict):
            if not training:
                return None
            transforms = []
            if config["aug_color_jitter"]:
                transforms.append(
                    A.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.1,
                        p=0.5,
                    )
                )
            if config["aug_compression"]:
                transforms.append(A.ImageCompression(quality_range=(50, 95), p=0.3))
            if config["aug_gauss_noise"]:
                transforms.append(A.GaussNoise(std_range=(0.02, 0.08), p=0.3))
            if config["aug_gauss_blur"]:
                transforms.append(A.GaussianBlur(blur_limit=(3, 5), p=0.2))
            return A.Compose(transforms) if transforms else None


        def rgb_to_tensor(image_rgb: np.ndarray) -> torch.Tensor:
            tensor = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float() / 255.0
            return (tensor - IMAGENET_MEAN) / IMAGENET_STD


        def denormalize_rgb(rgb_tensor: torch.Tensor) -> np.ndarray:
            rgb = rgb_tensor.detach().cpu().clone()
            rgb = rgb[:3] * IMAGENET_STD + IMAGENET_MEAN
            rgb = rgb.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
            return rgb


        def default_degradation(image_bgr: np.ndarray) -> np.ndarray:
            return image_bgr


        class TamperingDataset(Dataset):
            def __init__(self, pairs: list[dict], config: dict, training: bool = False, degradation_fn=None):
                self.pairs = list(pairs)
                self.config = config
                self.training = training
                self.degradation_fn = degradation_fn or default_degradation
                self.spatial_transform = build_spatial_transform(training, config)
                self.rgb_transform = build_rgb_only_transform(training, config)

            def __len__(self) -> int:
                return len(self.pairs)

            def _load_mask(self, pair: dict, height: int, width: int) -> np.ndarray:
                mask_path = pair.get("mask_path")
                if mask_path and Path(mask_path).exists():
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        mask = np.zeros((height, width), dtype=np.uint8)
                else:
                    mask = np.zeros((height, width), dtype=np.uint8)
                mask = (mask > 0).astype(np.uint8) * 255
                return mask

            def __getitem__(self, idx: int):
                pair = self.pairs[idx]
                image_bgr = cv2.imread(pair["image_path"], cv2.IMREAD_COLOR)
                if image_bgr is None:
                    raise FileNotFoundError(f"Could not read image: {pair['image_path']}")
                image_bgr = self.degradation_fn(image_bgr)
                height, width = image_bgr.shape[:2]

                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                mask = self._load_mask(pair, height, width)

                spatial = self.spatial_transform(image=image_rgb, mask=mask)
                image_rgb = spatial["image"]
                mask = spatial["mask"]

                if self.rgb_transform is not None:
                    image_rgb = self.rgb_transform(image=image_rgb)["image"]

                rgb_tensor = rgb_to_tensor(image_rgb)
                mask_tensor = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)
                label = torch.tensor(float(mask_tensor.sum().item() > 0), dtype=torch.float32)

                if self.config["use_ela"]:
                    ela_source_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                    ela_map = compute_ela(ela_source_bgr, self.config["ela_quality"])
                    ela_tensor = torch.from_numpy(ela_map.astype(np.float32) / 255.0).unsqueeze(0)
                    image_tensor = torch.cat([rgb_tensor, ela_tensor], dim=0)
                else:
                    image_tensor = rgb_tensor

                forgery_type = pair.get("forgery_type", "splicing" if label.item() else "authentic")
                return image_tensor, mask_tensor, label, forgery_type


        def build_loader(dataset: Dataset, config: dict, shuffle: bool) -> DataLoader:
            loader_kwargs = {
                "batch_size": config["batch_size"],
                "shuffle": shuffle,
                "drop_last": shuffle,
                "num_workers": config["num_workers"],
                "pin_memory": bool(config["pin_memory"] and DEVICE.type == "cuda"),
                "worker_init_fn": seed_worker,
            }
            if config["num_workers"] > 0:
                loader_kwargs["persistent_workers"] = bool(config["persistent_workers"])
                loader_kwargs["prefetch_factor"] = int(config["prefetch_factor"])
            return DataLoader(dataset, **loader_kwargs)


        def build_dataloaders(config: dict, splits: dict, test_degradation=None):
            train_dataset = TamperingDataset(splits["train"], config, training=True)
            val_dataset = TamperingDataset(splits["val"], config, training=False)
            test_dataset = TamperingDataset(splits["test"], config, training=False, degradation_fn=test_degradation)

            train_loader = build_loader(train_dataset, config, shuffle=True)
            val_loader = build_loader(val_dataset, config, shuffle=False)
            test_loader = build_loader(test_dataset, config, shuffle=False)
            return {
                "train_dataset": train_dataset,
                "val_dataset": val_dataset,
                "test_dataset": test_dataset,
                "train_loader": train_loader,
                "val_loader": val_loader,
                "test_loader": test_loader,
            }


        PREVIEW_LOADERS = build_dataloaders(CONFIG, SPLITS)
        preview_images, preview_masks, preview_labels, preview_types = next(iter(PREVIEW_LOADERS["train_loader"]))
        print(f"Preview image batch shape: {tuple(preview_images.shape)}")
        print(f"Preview mask batch shape: {tuple(preview_masks.shape)}")
        print(f"Preview labels shape: {tuple(preview_labels.shape)}")
        print(f"Preview forgery types: {list(preview_types)[:4]}")
        """
    )


def model_cell() -> dict:
    return code(
        """
        def replace_module(root_module: nn.Module, module_name: str, new_module: nn.Module) -> None:
            parts = module_name.split(".")
            parent = root_module
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_module)


        def inflate_first_conv(encoder: nn.Module, in_channels: int) -> None:
            if in_channels == 3:
                return
            first_conv_name = None
            first_conv = None
            for name, module in encoder.named_modules():
                if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                    first_conv_name = name
                    first_conv = module
                    break

            if first_conv is None or first_conv_name is None:
                raise RuntimeError("Could not find an encoder conv with 3 input channels.")

            new_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                dilation=first_conv.dilation,
                groups=first_conv.groups,
                bias=first_conv.bias is not None,
                padding_mode=first_conv.padding_mode,
            )

            with torch.no_grad():
                new_conv.weight[:, :3] = first_conv.weight
                mean_weight = first_conv.weight.mean(dim=1, keepdim=True)
                for channel_index in range(3, in_channels):
                    new_conv.weight[:, channel_index : channel_index + 1] = mean_weight
                if first_conv.bias is not None:
                    new_conv.bias.copy_(first_conv.bias)

            replace_module(encoder, first_conv_name, new_conv)


        def build_segmentation_model(config: dict) -> nn.Module:
            architecture_map = {
                "unet": smp.Unet,
                "deeplabv3plus": smp.DeepLabV3Plus,
            }
            architecture_name = config["architecture"].lower()
            if architecture_name not in architecture_map:
                raise ValueError(f"Unsupported architecture: {config['architecture']}")

            architecture_cls = architecture_map[architecture_name]
            model = architecture_cls(
                encoder_name=config["encoder_name"],
                encoder_weights="imagenet" if config["pretrained"] else None,
                in_channels=3,
                classes=1,
                activation=None,
            )

            if config["in_channels"] > 3:
                inflate_first_conv(model.encoder, config["in_channels"])

            return model


        class DualTaskModel(nn.Module):
            def __init__(self, config: dict):
                super().__init__()
                self.config = config
                self.segmentation_model = build_segmentation_model(config)
                encoder_channels = self.segmentation_model.encoder.out_channels[-1]
                self.cls_head = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Dropout(p=config.get("decoder_dropout", 0.3)),
                    nn.Linear(encoder_channels, 1),
                )
                self._cached_feature_map = None

            def forward(self, x: torch.Tensor):
                features = self.segmentation_model.encoder(x)
                self._cached_feature_map = features[-1]
                decoder_output = self.segmentation_model.decoder(*features)
                seg_logits = self.segmentation_model.segmentation_head(decoder_output)
                cls_logits = self.cls_head(features[-1])
                return seg_logits, cls_logits


        def build_model(config: dict) -> nn.Module:
            return DualTaskModel(config)


        def run_model_shape_smoke_test(base_config: dict) -> None:
            for architecture_name in ["unet", "deeplabv3plus"]:
                test_config = copy.deepcopy(base_config)
                test_config["architecture"] = architecture_name
                model = build_model(test_config)
                dummy = torch.randn(2, test_config["in_channels"], 128, 128)
                with torch.no_grad():
                    seg_logits, cls_logits = model(dummy)
                print(
                    architecture_name,
                    "segmentation shape:",
                    tuple(seg_logits.shape),
                    "classification shape:",
                    tuple(cls_logits.shape),
                )
                del model, dummy


        run_model_shape_smoke_test(CONFIG)
        """
    )


def loss_and_checkpoint_cell() -> dict:
    return code(
        """
        class BCEDiceLoss(nn.Module):
            def __init__(self, pos_weight: torch.Tensor | None = None):
                super().__init__()
                if pos_weight is not None:
                    self.register_buffer("pos_weight", pos_weight.float().view(1))
                else:
                    self.pos_weight = None

            def _bce(self, logits: torch.Tensor, targets: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
                pos_weight = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
                return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction=reduction)

            def per_sample(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
                bce_map = self._bce(logits, targets, reduction="none")
                bce_per_sample = bce_map.view(logits.size(0), -1).mean(dim=1)

                probs = torch.sigmoid(logits)
                probs = probs.view(logits.size(0), -1)
                targets = targets.view(targets.size(0), -1)
                intersection = (probs * targets).sum(dim=1)
                dice = (2.0 * intersection + 1.0) / (probs.sum(dim=1) + targets.sum(dim=1) + 1.0)
                dice_loss = 1.0 - dice
                return bce_per_sample + dice_loss

            def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, dict]:
                per_sample = self.per_sample(logits, targets)
                total = per_sample.mean()
                probs = torch.sigmoid(logits)
                probs = probs.view(logits.size(0), -1)
                targets_flat = targets.view(targets.size(0), -1)
                intersection = (probs * targets_flat).sum(dim=1)
                dice = (2.0 * intersection + 1.0) / (probs.sum(dim=1) + targets_flat.sum(dim=1) + 1.0)
                dice_loss = (1.0 - dice).mean()
                bce = self._bce(logits, targets, reduction="mean")
                return total, {"bce": bce.detach(), "dice": dice_loss.detach()}


        def compute_edge_mask(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            kernel = torch.ones((1, 1, kernel_size, kernel_size), device=mask.device)
            padding = kernel_size // 2
            dilated = F.conv2d(mask.float(), kernel, padding=padding)
            dilated = (dilated > 0).float()
            eroded = F.conv2d(mask.float(), kernel, padding=padding)
            eroded = (eroded >= kernel_size * kernel_size).float()
            edge = (dilated - eroded).clamp(0.0, 1.0)
            return edge


        class MultiTaskLoss(nn.Module):
            def __init__(self, seg_pos_weight=None, cls_pos_weight=None, config=None):
                super().__init__()
                config = config or CONFIG
                self.seg_loss = BCEDiceLoss(pos_weight=seg_pos_weight)
                self.cls_loss_weight = config["cls_loss_weight"]
                self.use_edge_loss = bool(config["use_edge_loss"])
                self.edge_loss_weight = float(config["edge_loss_weight"])
                self.edge_loss_lambda = float(config["edge_loss_lambda"])
                if cls_pos_weight is not None:
                    self.register_buffer("cls_pos_weight", cls_pos_weight.float().view(1))
                else:
                    self.cls_pos_weight = None

            def forward(self, seg_logits, cls_logits, seg_targets, cls_targets):
                seg_total, seg_parts = self.seg_loss(seg_logits, seg_targets)
                cls_pos_weight = self.cls_pos_weight.to(cls_logits.device) if self.cls_pos_weight is not None else None
                cls_loss = F.binary_cross_entropy_with_logits(
                    cls_logits.view(-1),
                    cls_targets.float().view(-1),
                    pos_weight=cls_pos_weight,
                )

                edge_loss = torch.tensor(0.0, device=seg_logits.device)
                if self.use_edge_loss:
                    edge_mask = compute_edge_mask(seg_targets)
                    edge_weight = 1.0 + self.edge_loss_weight * edge_mask
                    edge_loss = F.binary_cross_entropy_with_logits(
                        seg_logits,
                        seg_targets,
                        weight=edge_weight,
                    )

                total = seg_total + self.cls_loss_weight * cls_loss + self.edge_loss_lambda * edge_loss
                parts = {
                    "total_loss": total.detach(),
                    "seg_loss": seg_total.detach(),
                    "seg_bce": seg_parts["bce"],
                    "seg_dice": seg_parts["dice"],
                    "cls_loss": cls_loss.detach(),
                    "edge_loss": edge_loss.detach(),
                }
                return total, parts


        def compute_pos_weights(train_pairs: list[dict], config: dict) -> tuple[torch.Tensor, torch.Tensor]:
            fg_pixels = 0
            bg_pixels = 0
            tampered_count = 0
            authentic_count = 0
            for pair in tqdm(train_pairs, desc="Computing class weights", leave=False):
                if pair["label"] == 1 and pair.get("mask_path"):
                    mask = cv2.imread(pair["mask_path"], cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        continue
                    mask = (mask > 0).astype(np.uint8)
                    fg = int(mask.sum())
                    bg = int(mask.size - fg)
                    fg_pixels += fg
                    bg_pixels += bg
                    tampered_count += 1
                else:
                    image = cv2.imread(pair["image_path"], cv2.IMREAD_GRAYSCALE)
                    if image is not None:
                        bg_pixels += int(image.size)
                    authentic_count += 1

            fg_pixels = max(fg_pixels, 1)
            tampered_count = max(tampered_count, 1)
            authentic_count = max(authentic_count, 1)

            seg_pos_weight = torch.tensor([bg_pixels / fg_pixels], dtype=torch.float32)
            cls_pos_weight = torch.tensor([authentic_count / tampered_count], dtype=torch.float32)
            print(f"Segmentation pos_weight: {seg_pos_weight.item():.4f}")
            print(f"Classification pos_weight: {cls_pos_weight.item():.4f}")
            return seg_pos_weight, cls_pos_weight


        def build_optimizer(model: nn.Module, config: dict):
            encoder_params = list(model.segmentation_model.encoder.parameters())
            decoder_params = (
                list(model.segmentation_model.decoder.parameters())
                + list(model.segmentation_model.segmentation_head.parameters())
                + list(model.cls_head.parameters())
            )
            optimizer = torch.optim.AdamW(
                [
                    {"params": encoder_params, "lr": config["encoder_lr"]},
                    {"params": decoder_params, "lr": config["decoder_lr"]},
                ],
                weight_decay=config["weight_decay"],
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=config["scheduler_factor"],
                patience=config["scheduler_patience"],
                min_lr=config["min_lr"],
            )
            return optimizer, scheduler


        def save_checkpoint(path: Path, payload: dict) -> None:
            ensure_dir(path.parent)
            torch.save(payload, path)


        def build_checkpoint_payload(model, optimizer, scheduler, scaler, epoch, config, history, train_stats, val_stats):
            return {
                "epoch": epoch,
                "config": copy.deepcopy(config),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "history": copy.deepcopy(history),
                "train_stats": copy.deepcopy(train_stats),
                "val_stats": copy.deepcopy(val_stats),
            }


        def load_model_from_checkpoint(checkpoint_path: Path, config: dict, device: torch.device):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model = build_model(config).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            return model, checkpoint
        """
    )


def training_cell() -> dict:
    return code(
        """
        def safe_div(numerator: float, denominator: float) -> float:
            return float(numerator) / float(denominator) if denominator else 0.0


        def counts_to_metrics(tp: int, fp: int, fn: int) -> dict:
            precision = safe_div(tp, tp + fp)
            recall = safe_div(tp, tp + fn)
            f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
            iou = safe_div(tp, tp + fp + fn)
            return {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "iou": iou,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
            }


        def image_metrics(scores: list[float], labels: list[int], threshold_grid=None, fixed_threshold=None) -> dict:
            scores_array = np.asarray(scores, dtype=np.float32)
            labels_array = np.asarray(labels, dtype=np.int32)
            result = {
                "threshold": float(fixed_threshold) if fixed_threshold is not None else None,
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "auc": None,
                "average_precision": None,
            }

            if len(np.unique(labels_array)) > 1:
                result["auc"] = float(roc_auc_score(labels_array, scores_array))
                result["average_precision"] = float(average_precision_score(labels_array, scores_array))

            if fixed_threshold is None:
                threshold_grid = threshold_grid or np.arange(0.1, 0.9, 0.05)
                best_threshold = 0.5
                best_f1 = -1.0
                for threshold in threshold_grid:
                    preds = (scores_array >= threshold).astype(np.int32)
                    f1 = f1_score(labels_array, preds, zero_division=0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = float(threshold)
                result["threshold"] = best_threshold
            else:
                best_threshold = float(fixed_threshold)

            preds = (scores_array >= best_threshold).astype(np.int32)
            result["accuracy"] = float(accuracy_score(labels_array, preds))
            result["precision"] = float(precision_score(labels_array, preds, zero_division=0))
            result["recall"] = float(recall_score(labels_array, preds, zero_division=0))
            result["f1"] = float(f1_score(labels_array, preds, zero_division=0))
            return result


        def train_one_epoch(model, loader, criterion, optimizer, scaler, config, device, wandb_run=None):
            model.train()
            totals = defaultdict(float)
            type_loss_totals = defaultdict(float)
            type_loss_counts = defaultdict(int)
            optimizer.zero_grad(set_to_none=True)
            progress = tqdm(loader, desc="Train", leave=False)

            autocast_context = (
                torch.amp.autocast(device_type="cuda", enabled=AMP_ENABLED)
                if device.type == "cuda"
                else nullcontext()
            )

            for batch_index, (images, masks, labels, forgery_types) in enumerate(progress, start=1):
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with autocast_context:
                    seg_logits, cls_logits = model(images)
                    total_loss, loss_parts = criterion(seg_logits, cls_logits, masks, labels)
                    scaled_loss = total_loss / config["accumulation_steps"]

                scaler.scale(scaled_loss).backward()

                should_step = batch_index % config["accumulation_steps"] == 0 or batch_index == len(loader)
                grad_norm = None
                if should_step:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                totals["loss"] += float(total_loss.item())
                for key, value in loss_parts.items():
                    totals[key] += float(value.item())

                if grad_norm is not None:
                    totals["grad_norm"] += float(grad_norm.item())

                if config.get("track_forgery_type_loss", False):
                    with torch.no_grad():
                        per_sample_seg = criterion.seg_loss.per_sample(seg_logits.detach(), masks.detach()).cpu().numpy()
                    for sample_loss, forgery_type in zip(per_sample_seg, forgery_types):
                        type_loss_totals[str(forgery_type)] += float(sample_loss)
                        type_loss_counts[str(forgery_type)] += 1

                progress.set_postfix(
                    loss=f"{totals['loss'] / batch_index:.4f}",
                    seg=f"{totals['seg_loss'] / batch_index:.4f}",
                    cls=f"{totals['cls_loss'] / batch_index:.4f}",
                )

            stats = {key: value / max(len(loader), 1) for key, value in totals.items()}
            stats["per_forgery_type_seg_loss"] = {
                forgery_type: type_loss_totals[forgery_type] / max(type_loss_counts[forgery_type], 1)
                for forgery_type in sorted(type_loss_totals)
            }

            if wandb_run is not None:
                log_payload = {f"train/{key}": value for key, value in stats.items() if not isinstance(value, dict)}
                for forgery_type, loss_value in stats["per_forgery_type_seg_loss"].items():
                    log_payload[f"train/per_type_seg_loss/{forgery_type}"] = loss_value
                wandb_run.log(log_payload)

            return stats


        def validate_one_epoch(model, loader, criterion, config, device, wandb_run=None):
            model.eval()
            totals = defaultdict(float)
            threshold_counts = {float(t): {"tp": 0, "fp": 0, "fn": 0} for t in config["thresholds"]}
            image_scores = []
            image_labels = []

            autocast_context = (
                torch.amp.autocast(device_type="cuda", enabled=AMP_ENABLED)
                if device.type == "cuda"
                else nullcontext()
            )

            with torch.no_grad():
                for images, masks, labels, _forgery_types in tqdm(loader, desc="Validate", leave=False):
                    images = images.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    with autocast_context:
                        seg_logits, cls_logits = model(images)
                        total_loss, loss_parts = criterion(seg_logits, cls_logits, masks, labels)

                    totals["loss"] += float(total_loss.item())
                    for key, value in loss_parts.items():
                        totals[key] += float(value.item())

                    probs = torch.sigmoid(seg_logits).detach().cpu().numpy()
                    masks_np = masks.detach().cpu().numpy()
                    labels_np = labels.detach().cpu().numpy().astype(np.int32)
                    cls_scores = torch.sigmoid(cls_logits).view(-1).detach().cpu().numpy()

                    image_scores.extend(cls_scores.tolist())
                    image_labels.extend(labels_np.tolist())

                    tampered_indices = np.where(labels_np == 1)[0]
                    if len(tampered_indices) == 0:
                        continue

                    tampered_probs = probs[tampered_indices]
                    tampered_targets = masks_np[tampered_indices]
                    for threshold in config["thresholds"]:
                        preds = (tampered_probs >= threshold).astype(np.uint8)
                        targets = (tampered_targets >= 0.5).astype(np.uint8)
                        threshold_counts[float(threshold)]["tp"] += int(np.logical_and(preds == 1, targets == 1).sum())
                        threshold_counts[float(threshold)]["fp"] += int(np.logical_and(preds == 1, targets == 0).sum())
                        threshold_counts[float(threshold)]["fn"] += int(np.logical_and(preds == 0, targets == 1).sum())

            averaged = {key: value / max(len(loader), 1) for key, value in totals.items()}
            threshold_metrics = {}
            best_f1 = {"threshold": None, "metrics": None}
            best_iou = {"threshold": None, "metrics": None}
            for threshold, counts in threshold_counts.items():
                metrics = counts_to_metrics(counts["tp"], counts["fp"], counts["fn"])
                threshold_metrics[threshold] = metrics
                if best_f1["metrics"] is None or metrics["f1"] > best_f1["metrics"]["f1"]:
                    best_f1 = {"threshold": float(threshold), "metrics": metrics}
                if best_iou["metrics"] is None or metrics["iou"] > best_iou["metrics"]["iou"]:
                    best_iou = {"threshold": float(threshold), "metrics": metrics}

            img_metrics = image_metrics(image_scores, image_labels, threshold_grid=config["thresholds"])
            summary = {
                "loss": averaged["loss"],
                "seg_loss": averaged["seg_loss"],
                "cls_loss": averaged["cls_loss"],
                "edge_loss": averaged["edge_loss"],
                "best_pixel_threshold_f1": best_f1["threshold"],
                "best_pixel_threshold_iou": best_iou["threshold"],
                "best_pixel_f1": best_f1["metrics"]["f1"],
                "best_pixel_iou": best_iou["metrics"]["iou"],
                "best_pixel_precision": best_f1["metrics"]["precision"],
                "best_pixel_recall": best_f1["metrics"]["recall"],
                "threshold_metrics": threshold_metrics,
                "image_metrics": img_metrics,
            }

            if wandb_run is not None:
                log_payload = {
                    "val/loss": summary["loss"],
                    "val/seg_loss": summary["seg_loss"],
                    "val/cls_loss": summary["cls_loss"],
                    "val/edge_loss": summary["edge_loss"],
                    "val/best_pixel_f1": summary["best_pixel_f1"],
                    "val/best_pixel_iou": summary["best_pixel_iou"],
                    "val/best_pixel_threshold_f1": summary["best_pixel_threshold_f1"],
                    "val/best_pixel_threshold_iou": summary["best_pixel_threshold_iou"],
                    "val/image_auc": summary["image_metrics"]["auc"],
                    "val/image_f1": summary["image_metrics"]["f1"],
                }
                wandb_run.log(log_payload)

            return summary


        def fit_model(experiment_config: dict, run_dir: Path, splits: dict, run_name: str):
            set_seed(experiment_config["seed"])
            run_dir = ensure_dir(run_dir)
            ckpt_dir = ensure_dir(run_dir / "checkpoints")
            plot_dir = ensure_dir(run_dir / "plots")
            report_dir = ensure_dir(run_dir / "reports")
            save_json(run_dir / "config.json", experiment_config)

            run = start_wandb_run(experiment_config, run_name)
            loaders = build_dataloaders(experiment_config, splits)
            seg_pos_weight, cls_pos_weight = compute_pos_weights(splits["train"], experiment_config)
            model = build_model(experiment_config).to(DEVICE)
            criterion = MultiTaskLoss(seg_pos_weight=seg_pos_weight, cls_pos_weight=cls_pos_weight, config=experiment_config).to(DEVICE)
            optimizer, scheduler = build_optimizer(model, experiment_config)
            scaler = GradScaler(enabled=AMP_ENABLED)

            history = []
            best_primary = -1.0
            best_iou = -1.0
            best_primary_path = ckpt_dir / "best_model_primary.pt"
            best_iou_path = ckpt_dir / "best_model_iou.pt"
            final_model_path = ckpt_dir / "final_model.pt"
            last_checkpoint_path = ckpt_dir / "last_checkpoint.pt"
            epochs_without_primary_improvement = 0

            for epoch in range(1, experiment_config["max_epochs"] + 1):
                epoch_start = time.time()
                train_stats = train_one_epoch(model, loaders["train_loader"], criterion, optimizer, scaler, experiment_config, DEVICE, run)
                val_stats = validate_one_epoch(model, loaders["val_loader"], criterion, experiment_config, DEVICE, run)
                scheduler.step(val_stats["best_pixel_f1"])

                epoch_summary = {
                    "epoch": epoch,
                    "elapsed_sec": time.time() - epoch_start,
                    "train": train_stats,
                    "val": val_stats,
                    "lr_encoder": optimizer.param_groups[0]["lr"],
                    "lr_decoder": optimizer.param_groups[1]["lr"],
                }
                history.append(epoch_summary)

                checkpoint_payload = build_checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    experiment_config,
                    history,
                    train_stats,
                    val_stats,
                )

                save_checkpoint(last_checkpoint_path, checkpoint_payload)
                if epoch % experiment_config["save_every_n_epochs"] == 0:
                    save_checkpoint(ckpt_dir / f"checkpoint_epoch_{epoch}.pt", checkpoint_payload)

                if val_stats["best_pixel_f1"] > best_primary:
                    best_primary = val_stats["best_pixel_f1"]
                    save_checkpoint(best_primary_path, checkpoint_payload)
                    epochs_without_primary_improvement = 0
                else:
                    epochs_without_primary_improvement += 1

                if val_stats["best_pixel_iou"] > best_iou:
                    best_iou = val_stats["best_pixel_iou"]
                    save_checkpoint(best_iou_path, checkpoint_payload)

                if run is not None:
                    run.log(
                        {
                            "epoch": epoch,
                            "lr/encoder": optimizer.param_groups[0]["lr"],
                            "lr/decoder": optimizer.param_groups[1]["lr"],
                            "time/epoch_sec": epoch_summary["elapsed_sec"],
                        }
                    )

                print(
                    f"Epoch {epoch:02d} | "
                    f"train_loss={train_stats['loss']:.4f} | "
                    f"val_f1={val_stats['best_pixel_f1']:.4f} | "
                    f"val_iou={val_stats['best_pixel_iou']:.4f} | "
                    f"pix_thr={val_stats['best_pixel_threshold_f1']:.2f} | "
                    f"img_thr={val_stats['image_metrics']['threshold']:.2f}"
                )

                if epochs_without_primary_improvement >= experiment_config["early_stopping_patience"]:
                    print(
                        f"Early stopping at epoch {epoch} after "
                        f"{epochs_without_primary_improvement} epochs without {experiment_config['primary_selection_metric']} improvement."
                    )
                    break

            final_payload = build_checkpoint_payload(
                model,
                optimizer,
                scheduler,
                scaler,
                history[-1]["epoch"],
                experiment_config,
                history,
                history[-1]["train"],
                history[-1]["val"],
            )
            save_checkpoint(final_model_path, final_payload)
            save_json(report_dir / "training_history.json", {"history": history})

            finish_wandb_run(run)
            return {
                "history": history,
                "paths": {
                    "best_model_primary": str(best_primary_path),
                    "best_model_f1": str(best_primary_path),
                    "best_model_iou": str(best_iou_path),
                    "last_checkpoint": str(last_checkpoint_path),
                    "final_model": str(final_model_path),
                    "plot_dir": str(plot_dir),
                    "report_dir": str(report_dir),
                    "ckpt_dir": str(ckpt_dir),
                },
                "config": copy.deepcopy(experiment_config),
                "loaders": loaders,
            }
        """
    )


def evaluation_and_visuals_cell() -> dict:
    return code(
        """
        def binary_metrics_from_masks(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
            pred_mask = pred_mask.astype(bool)
            gt_mask = gt_mask.astype(bool)
            tp = int(np.logical_and(pred_mask, gt_mask).sum())
            fp = int(np.logical_and(pred_mask, ~gt_mask).sum())
            fn = int(np.logical_and(~pred_mask, gt_mask).sum())
            tn = int(np.logical_and(~pred_mask, ~gt_mask).sum())

            if gt_mask.sum() == 0 and pred_mask.sum() == 0:
                return {
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                    "iou": 1.0,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                }

            metrics = counts_to_metrics(tp, fp, fn)
            metrics["tn"] = tn
            return metrics


        def boundary_f1_score(pred_mask: np.ndarray, gt_mask: np.ndarray, tolerance: int = 2) -> float:
            pred_boundary = find_boundaries(pred_mask.astype(bool), mode="inner")
            gt_boundary = find_boundaries(gt_mask.astype(bool), mode="inner")
            structure = np.ones((2 * tolerance + 1, 2 * tolerance + 1), dtype=np.uint8)
            pred_dilated = binary_dilation(pred_boundary, structure=structure)
            gt_dilated = binary_dilation(gt_boundary, structure=structure)

            if pred_boundary.sum() == 0:
                precision = 1.0 if gt_boundary.sum() == 0 else 0.0
            else:
                precision = float(np.logical_and(pred_boundary, gt_dilated).sum()) / float(pred_boundary.sum())

            if gt_boundary.sum() == 0:
                recall = 1.0 if pred_boundary.sum() == 0 else 0.0
            else:
                recall = float(np.logical_and(gt_boundary, pred_dilated).sum()) / float(gt_boundary.sum())

            if precision + recall == 0:
                return 0.0
            return 2.0 * precision * recall / (precision + recall)


        def mask_size_bucket(mask_ratio: float) -> str:
            if mask_ratio < 0.02:
                return "tiny"
            if mask_ratio < 0.05:
                return "small"
            if mask_ratio < 0.15:
                return "medium"
            return "large"


        def evaluate_model_detailed(
            model,
            loader,
            device,
            config,
            pixel_threshold: float,
            image_threshold: float | None = None,
            criterion=None,
            collect_pr: bool = True,
        ) -> dict:
            model.eval()
            tampered_counts = {"tp": 0, "fp": 0, "fn": 0}
            overall_counts = {"tp": 0, "fp": 0, "fn": 0}
            per_type_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
            size_bucket_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
            per_image_rows = []
            boundary_scores = []
            image_scores = []
            image_labels = []
            sampled_pixel_scores = []
            sampled_pixel_labels = []
            loss_totals = defaultdict(float)
            sample_index = 0

            autocast_context = (
                torch.amp.autocast(device_type="cuda", enabled=AMP_ENABLED)
                if device.type == "cuda"
                else nullcontext()
            )

            with torch.no_grad():
                for images, masks, labels, forgery_types in tqdm(loader, desc="Evaluate", leave=False):
                    images = images.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    with autocast_context:
                        seg_logits, cls_logits = model(images)
                        if criterion is not None:
                            total_loss, parts = criterion(seg_logits, cls_logits, masks, labels)
                            loss_totals["loss"] += float(total_loss.item())
                            for key, value in parts.items():
                                loss_totals[key] += float(value.item())

                    probs = torch.sigmoid(seg_logits).detach().cpu().numpy()
                    masks_np = masks.detach().cpu().numpy()
                    labels_np = labels.detach().cpu().numpy().astype(np.int32)
                    cls_scores = torch.sigmoid(cls_logits).view(-1).detach().cpu().numpy()
                    image_scores.extend(cls_scores.tolist())
                    image_labels.extend(labels_np.tolist())

                    tampered_indices = np.where(labels_np == 1)[0]
                    if collect_pr and len(sampled_pixel_scores) < config["max_pixel_pr_samples"] and len(tampered_indices) > 0:
                        flat_probs = probs[tampered_indices].reshape(-1)
                        flat_targets = masks_np[tampered_indices].reshape(-1)
                        remaining = config["max_pixel_pr_samples"] - len(sampled_pixel_scores)
                        take = min(remaining, min(50000, len(flat_probs)))
                        indices = np.random.choice(len(flat_probs), size=take, replace=False)
                        sampled_pixel_scores.extend(flat_probs[indices].tolist())
                        sampled_pixel_labels.extend(flat_targets[indices].tolist())

                    preds = (probs >= pixel_threshold).astype(np.uint8)
                    targets = (masks_np >= 0.5).astype(np.uint8)

                    for item_index in range(len(labels_np)):
                        pred_mask = preds[item_index, 0]
                        gt_mask = targets[item_index, 0]
                        metrics = binary_metrics_from_masks(pred_mask, gt_mask)

                        overall_counts["tp"] += metrics["tp"]
                        overall_counts["fp"] += metrics["fp"]
                        overall_counts["fn"] += metrics["fn"]

                        label = int(labels_np[item_index])
                        forgery_type = str(forgery_types[item_index])
                        if label == 1:
                            tampered_counts["tp"] += metrics["tp"]
                            tampered_counts["fp"] += metrics["fp"]
                            tampered_counts["fn"] += metrics["fn"]
                            per_type_counts[forgery_type]["tp"] += metrics["tp"]
                            per_type_counts[forgery_type]["fp"] += metrics["fp"]
                            per_type_counts[forgery_type]["fn"] += metrics["fn"]

                            ratio = float(gt_mask.sum()) / float(gt_mask.size)
                            bucket = mask_size_bucket(ratio)
                            size_bucket_counts[bucket]["tp"] += metrics["tp"]
                            size_bucket_counts[bucket]["fp"] += metrics["fp"]
                            size_bucket_counts[bucket]["fn"] += metrics["fn"]
                            boundary = boundary_f1_score(pred_mask, gt_mask, tolerance=config["boundary_tolerance"])
                            boundary_scores.append(boundary)
                        else:
                            ratio = 0.0
                            bucket = "authentic"
                            boundary = None

                        per_image_rows.append(
                            {
                                "dataset_index": sample_index,
                                "label": label,
                                "forgery_type": forgery_type,
                                "mask_ratio": ratio,
                                "size_bucket": bucket,
                                "pixel_precision": metrics["precision"],
                                "pixel_recall": metrics["recall"],
                                "pixel_f1": metrics["f1"],
                                "pixel_iou": metrics["iou"],
                                "boundary_f1": boundary,
                                "cls_score": float(cls_scores[item_index]),
                                "predicted_positive_pixels": int(pred_mask.sum()),
                                "gt_positive_pixels": int(gt_mask.sum()),
                            }
                        )
                        sample_index += 1

            image_summary = image_metrics(image_scores, image_labels, fixed_threshold=image_threshold)
            if image_threshold is None:
                image_threshold = image_summary["threshold"]

            result = {
                "pixel_threshold": float(pixel_threshold),
                "image_threshold": float(image_threshold),
                "overall": counts_to_metrics(**overall_counts),
                "tampered_only": counts_to_metrics(**tampered_counts),
                "per_forgery_type": {key: counts_to_metrics(**value) for key, value in per_type_counts.items()},
                "per_mask_size": {key: counts_to_metrics(**value) for key, value in size_bucket_counts.items()},
                "boundary_f1_mean": float(np.mean(boundary_scores)) if boundary_scores else None,
                "image_level": image_summary,
                "per_image": per_image_rows,
                "image_scores": image_scores,
                "image_labels": image_labels,
                "pixel_pr_samples": {
                    "scores": sampled_pixel_scores,
                    "labels": sampled_pixel_labels,
                },
                "loss": {key: value / max(len(loader), 1) for key, value in loss_totals.items()},
            }
            return result


        def plot_training_curves(history: list[dict], output_path: Path) -> None:
            history_df = pd.DataFrame(
                [
                    {
                        "epoch": row["epoch"],
                        "train_loss": row["train"]["loss"],
                        "val_loss": row["val"]["loss"],
                        "val_best_pixel_f1": row["val"]["best_pixel_f1"],
                        "val_best_pixel_iou": row["val"]["best_pixel_iou"],
                        "val_image_auc": row["val"]["image_metrics"]["auc"],
                    }
                    for row in history
                ]
            )
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train")
            axes[0].plot(history_df["epoch"], history_df["val_loss"], label="val")
            axes[0].set_title("Loss")
            axes[0].legend()

            axes[1].plot(history_df["epoch"], history_df["val_best_pixel_f1"], label="val F1")
            axes[1].plot(history_df["epoch"], history_df["val_best_pixel_iou"], label="val IoU")
            axes[1].set_title("Tampered-only Validation Metrics")
            axes[1].legend()

            axes[2].plot(history_df["epoch"], history_df["val_image_auc"], label="image AUC")
            axes[2].set_title("Image-level AUC")
            axes[2].legend()

            for ax in axes:
                ax.set_xlabel("Epoch")

            plt.tight_layout()
            plt.savefig(output_path, bbox_inches="tight")
            plt.close(fig)


        def plot_threshold_curve(threshold_metrics: dict, output_path: Path) -> None:
            thresholds = sorted(float(key) for key in threshold_metrics)
            f1_values = [threshold_metrics[threshold]["f1"] for threshold in thresholds]
            iou_values = [threshold_metrics[threshold]["iou"] for threshold in thresholds]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(thresholds, f1_values, label="F1")
            ax.plot(thresholds, iou_values, label="IoU")
            ax.set_xlabel("Threshold")
            ax.set_ylabel("Score")
            ax.set_title("Validation Threshold Sweep")
            ax.legend()
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches="tight")
            plt.close(fig)
        """
    )


def experiment_cell() -> dict:
    return code(
        """
        def maybe_float(value):
            return None if value is None else float(value)


        def pr_curve_payload(scores: list[float], labels: list[int]) -> dict:
            if not scores or len(np.unique(labels)) < 2:
                return {"precision": [], "recall": [], "thresholds": [], "average_precision": None}
            precision, recall, thresholds = precision_recall_curve(labels, scores)
            average_precision = average_precision_score(labels, scores)
            return {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "thresholds": thresholds.tolist(),
                "average_precision": float(average_precision),
            }


        def plot_pr_curves(evaluation_result: dict, output_path: Path) -> None:
            pixel_pr = pr_curve_payload(
                evaluation_result["pixel_pr_samples"]["scores"],
                evaluation_result["pixel_pr_samples"]["labels"],
            )
            image_pr = pr_curve_payload(
                evaluation_result["image_scores"],
                evaluation_result["image_labels"],
            )

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            if pixel_pr["precision"]:
                axes[0].plot(pixel_pr["recall"], pixel_pr["precision"])
                axes[0].set_title(f"Pixel PR (AP={pixel_pr['average_precision']:.4f})")
            else:
                axes[0].text(0.5, 0.5, "Insufficient class variation", ha="center", va="center")
                axes[0].set_title("Pixel PR")
            axes[0].set_xlabel("Recall")
            axes[0].set_ylabel("Precision")

            if image_pr["precision"]:
                axes[1].plot(image_pr["recall"], image_pr["precision"])
                axes[1].set_title(f"Image PR (AP={image_pr['average_precision']:.4f})")
            else:
                axes[1].text(0.5, 0.5, "Insufficient class variation", ha="center", va="center")
                axes[1].set_title("Image PR")
            axes[1].set_xlabel("Recall")
            axes[1].set_ylabel("Precision")

            plt.tight_layout()
            plt.savefig(output_path, bbox_inches="tight")
            plt.close(fig)


        def make_overlay(rgb: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
            overlay = rgb.copy()
            pred_mask = pred_mask.astype(bool)
            gt_mask = gt_mask.astype(bool)
            overlay[pred_mask, 0] = np.clip(overlay[pred_mask, 0] * 0.4 + 0.6, 0.0, 1.0)
            overlay[pred_mask, 1] = np.clip(overlay[pred_mask, 1] * 0.4, 0.0, 1.0)
            overlay[gt_mask, 1] = np.clip(overlay[gt_mask, 1] * 0.4 + 0.6, 0.0, 1.0)
            overlay[gt_mask, 0] = np.clip(overlay[gt_mask, 0] * 0.4, 0.0, 1.0)
            return overlay


        def select_visualization_rows(per_image_rows: list[dict], num_examples: int) -> pd.DataFrame:
            df = pd.DataFrame(per_image_rows)
            if df.empty:
                return df

            selections = []

            copymove_failures = (
                df[(df["label"] == 1) & (df["forgery_type"].str.contains("copy", na=False))]
                .sort_values("pixel_f1")
                .head(max(1, num_examples // 3))
            )
            selections.extend(copymove_failures["dataset_index"].tolist())

            tiny_failures = (
                df[(df["label"] == 1) & (df["size_bucket"] == "tiny")]
                .sort_values("pixel_f1")
                .head(max(1, num_examples // 3))
            )
            selections.extend(tiny_failures["dataset_index"].tolist())

            authentic_false_positives = (
                df[df["label"] == 0]
                .sort_values(["predicted_positive_pixels", "cls_score"], ascending=[False, False])
                .head(max(1, num_examples // 3))
            )
            selections.extend(authentic_false_positives["dataset_index"].tolist())

            low_f1_fill = (
                df[df["label"] == 1]
                .sort_values(["pixel_f1", "mask_ratio"], ascending=[True, True])
                .head(num_examples * 2)
            )
            selections.extend(low_f1_fill["dataset_index"].tolist())

            ordered_unique = []
            seen = set()
            for index in selections:
                if index not in seen:
                    ordered_unique.append(index)
                    seen.add(index)
                if len(ordered_unique) >= num_examples:
                    break

            return df[df["dataset_index"].isin(ordered_unique)].copy()


        def plot_qualitative_grid(
            model,
            dataset,
            device,
            pixel_threshold: float,
            per_image_rows: list[dict],
            output_path: Path,
            num_examples: int = 6,
        ) -> None:
            chosen_df = select_visualization_rows(per_image_rows, num_examples)
            if chosen_df.empty:
                return

            model.eval()
            columns = ["Original", "ELA", "Ground Truth", "Prediction", "Overlay"]
            fig, axes = plt.subplots(len(chosen_df), len(columns), figsize=(18, 3.6 * len(chosen_df)))
            if len(chosen_df) == 1:
                axes = np.expand_dims(axes, axis=0)

            with torch.no_grad():
                for row_index, (_, row) in enumerate(chosen_df.iterrows()):
                    image_tensor, mask_tensor, _label, _forgery_type = dataset[int(row["dataset_index"])]
                    image_batch = image_tensor.unsqueeze(0).to(device)
                    seg_logits, cls_logits = model(image_batch)
                    prob_map = torch.sigmoid(seg_logits)[0, 0].detach().cpu().numpy()
                    pred_mask = (prob_map >= pixel_threshold).astype(np.uint8)
                    gt_mask = (mask_tensor[0].numpy() >= 0.5).astype(np.uint8)
                    rgb = denormalize_rgb(image_tensor)
                    ela_map = image_tensor[3].detach().cpu().numpy() if image_tensor.shape[0] > 3 else np.zeros_like(gt_mask, dtype=np.float32)
                    overlay = make_overlay(rgb, gt_mask, pred_mask)
                    cls_score = float(torch.sigmoid(cls_logits).view(-1)[0].item())

                    axes[row_index, 0].imshow(rgb)
                    axes[row_index, 1].imshow(ela_map, cmap="hot")
                    axes[row_index, 2].imshow(gt_mask, cmap="gray")
                    axes[row_index, 3].imshow(pred_mask, cmap="gray")
                    axes[row_index, 4].imshow(overlay)

                    row_title = (
                        f"idx={int(row['dataset_index'])} | type={row['forgery_type']} | "
                        f"f1={row['pixel_f1']:.3f} | cls={cls_score:.3f}"
                    )
                    axes[row_index, 0].set_ylabel(row_title, fontsize=9)
                    for col_index, column_name in enumerate(columns):
                        axes[row_index, col_index].axis("off")
                        if row_index == 0:
                            axes[row_index, col_index].set_title(column_name)

            plt.tight_layout()
            plt.savefig(output_path, bbox_inches="tight")
            plt.close(fig)


        def jpeg_degradation(quality: int):
            def _apply(image_bgr: np.ndarray) -> np.ndarray:
                success, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
                if not success:
                    return image_bgr
                decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                return decoded if decoded is not None else image_bgr

            return _apply


        def gaussian_noise_degradation(std_ratio: float):
            def _apply(image_bgr: np.ndarray) -> np.ndarray:
                noise = np.random.normal(0.0, std_ratio * 255.0, size=image_bgr.shape).astype(np.float32)
                degraded = np.clip(image_bgr.astype(np.float32) + noise, 0.0, 255.0)
                return degraded.astype(np.uint8)

            return _apply


        def blur_degradation(kernel_size: int):
            def _apply(image_bgr: np.ndarray) -> np.ndarray:
                return cv2.GaussianBlur(image_bgr, (kernel_size, kernel_size), 0)

            return _apply


        def resize_restore_degradation(scale: float):
            def _apply(image_bgr: np.ndarray) -> np.ndarray:
                height, width = image_bgr.shape[:2]
                resized = cv2.resize(
                    image_bgr,
                    (max(1, int(width * scale)), max(1, int(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
                return cv2.resize(resized, (width, height), interpolation=cv2.INTER_LINEAR)

            return _apply


        def build_degradation_map() -> dict:
            return {
                "clean": default_degradation,
                "jpeg_qf70": jpeg_degradation(70),
                "jpeg_qf50": jpeg_degradation(50),
                "gaussian_noise_light": gaussian_noise_degradation(0.03),
                "gaussian_noise_heavy": gaussian_noise_degradation(0.06),
                "gaussian_blur": blur_degradation(5),
                "resize_0.75x": resize_restore_degradation(0.75),
                "resize_0.5x": resize_restore_degradation(0.5),
            }


        def run_mask_randomization_test(
            model,
            base_dataset,
            device,
            config,
            pixel_threshold: float,
            image_threshold: float,
            iterations: int = 5,
        ) -> dict:
            baseline_loader = build_loader(base_dataset, config, shuffle=False)
            baseline_result = evaluate_model_detailed(
                model,
                baseline_loader,
                device,
                config,
                pixel_threshold=pixel_threshold,
                image_threshold=image_threshold,
                collect_pr=False,
            )

            shuffled_rows = []
            tampered_mask_paths = [pair["mask_path"] for pair in base_dataset.pairs if pair["label"] == 1 and pair.get("mask_path")]
            for iteration in range(iterations):
                shuffled_pairs = copy.deepcopy(base_dataset.pairs)
                shuffled_mask_paths = list(tampered_mask_paths)
                random.Random(config["seed"] + iteration).shuffle(shuffled_mask_paths)
                shuffled_iter = iter(shuffled_mask_paths)
                for pair in shuffled_pairs:
                    if pair["label"] == 1 and pair.get("mask_path"):
                        pair["mask_path"] = next(shuffled_iter)

                shuffled_dataset = TamperingDataset(
                    shuffled_pairs,
                    config,
                    training=False,
                    degradation_fn=base_dataset.degradation_fn,
                )
                shuffled_loader = build_loader(shuffled_dataset, config, shuffle=False)
                shuffled_result = evaluate_model_detailed(
                    model,
                    shuffled_loader,
                    device,
                    config,
                    pixel_threshold=pixel_threshold,
                    image_threshold=image_threshold,
                    collect_pr=False,
                )
                shuffled_rows.append(
                    {
                        "iteration": int(iteration),
                        "tampered_only_f1": float(shuffled_result["tampered_only"]["f1"]),
                        "tampered_only_iou": float(shuffled_result["tampered_only"]["iou"]),
                    }
                )

            mean_shuffled_f1 = float(np.mean([row["tampered_only_f1"] for row in shuffled_rows])) if shuffled_rows else 0.0
            return {
                "baseline_tampered_only_f1": float(baseline_result["tampered_only"]["f1"]),
                "mean_shuffled_tampered_only_f1": mean_shuffled_f1,
                "drop": float(baseline_result["tampered_only"]["f1"] - mean_shuffled_f1),
                "iterations": shuffled_rows,
            }


        def run_robustness_suite(
            model,
            base_config: dict,
            splits: dict,
            device,
            pixel_threshold: float,
            image_threshold: float,
        ) -> tuple[pd.DataFrame, dict]:
            degradation_map = build_degradation_map()
            rows = []
            for name in base_config["robustness_degradations"]:
                degradation_fn = degradation_map[name]
                loaders = build_dataloaders(base_config, splits, test_degradation=degradation_fn)
                result = evaluate_model_detailed(
                    model,
                    loaders["test_loader"],
                    device,
                    base_config,
                    pixel_threshold=pixel_threshold,
                    image_threshold=image_threshold,
                    collect_pr=False,
                )
                rows.append(
                    {
                        "condition": name,
                        "tampered_only_f1": float(result["tampered_only"]["f1"]),
                        "tampered_only_iou": float(result["tampered_only"]["iou"]),
                        "copymove_f1": maybe_float(result["per_forgery_type"].get("copymove", {}).get("f1")),
                        "splicing_f1": maybe_float(result["per_forgery_type"].get("splicing", {}).get("f1")),
                        "boundary_f1_mean": maybe_float(result["boundary_f1_mean"]),
                        "image_auc": maybe_float(result["image_level"].get("auc")),
                    }
                )

            robustness_df = pd.DataFrame(rows)
            clean_f1 = float(robustness_df.loc[robustness_df["condition"] == "clean", "tampered_only_f1"].iloc[0])
            jpeg_qf50_f1 = float(robustness_df.loc[robustness_df["condition"] == "jpeg_qf50", "tampered_only_f1"].iloc[0])
            report = {
                "robustness_indicator_jpeg_qf50": float(clean_f1 - jpeg_qf50_f1),
                "rows": robustness_df.to_dict("records"),
            }
            return robustness_df, report


        def build_eval_criterion(experiment_config: dict, splits: dict):
            seg_pos_weight, cls_pos_weight = compute_pos_weights(splits["train"], experiment_config)
            return MultiTaskLoss(seg_pos_weight=seg_pos_weight, cls_pos_weight=cls_pos_weight, config=experiment_config).to(DEVICE)


        def summarize_result_row(label: str, evaluation_result: dict) -> dict:
            return {
                "experiment": label,
                "tampered_only_f1": float(evaluation_result["tampered_only"]["f1"]),
                "tampered_only_iou": float(evaluation_result["tampered_only"]["iou"]),
                "boundary_f1_mean": maybe_float(evaluation_result["boundary_f1_mean"]),
                "image_auc": maybe_float(evaluation_result["image_level"].get("auc")),
                "image_f1": maybe_float(evaluation_result["image_level"].get("f1")),
                "pixel_threshold": float(evaluation_result["pixel_threshold"]),
                "image_threshold": float(evaluation_result["image_threshold"]),
                "copymove_f1": maybe_float(evaluation_result["per_forgery_type"].get("copymove", {}).get("f1")),
                "splicing_f1": maybe_float(evaluation_result["per_forgery_type"].get("splicing", {}).get("f1")),
            }


        def mean_std_summary(rows: list[dict], metrics: list[str]) -> dict:
            summary = {}
            rows_df = pd.DataFrame(rows)
            for metric in metrics:
                values = rows_df[metric].dropna().astype(float)
                summary[metric] = {
                    "mean": float(values.mean()) if not values.empty else None,
                    "std": float(values.std(ddof=0)) if not values.empty else None,
                }
            return summary


        def train_and_evaluate_once(
            experiment_config: dict,
            splits: dict,
            run_dir: Path,
            run_name: str,
            save_visuals: bool = True,
            save_robustness: bool = True,
            save_mask_randomization: bool = True,
            save_pr: bool = True,
        ) -> dict:
            artifacts = fit_model(experiment_config, run_dir, splits, run_name)
            report_dir = ensure_dir(Path(artifacts["paths"]["report_dir"]))
            plot_dir = ensure_dir(Path(artifacts["paths"]["plot_dir"]))
            checkpoint_path = Path(artifacts["paths"]["best_model_primary"])
            model, checkpoint = load_model_from_checkpoint(checkpoint_path, experiment_config, DEVICE)
            criterion = build_eval_criterion(experiment_config, splits)
            pixel_threshold = float(checkpoint["val_stats"]["best_pixel_threshold_f1"])
            image_threshold = float(checkpoint["val_stats"]["image_metrics"]["threshold"])

            val_result = evaluate_model_detailed(
                model,
                artifacts["loaders"]["val_loader"],
                DEVICE,
                experiment_config,
                pixel_threshold=pixel_threshold,
                image_threshold=image_threshold,
                criterion=criterion,
                collect_pr=False,
            )
            test_result = evaluate_model_detailed(
                model,
                artifacts["loaders"]["test_loader"],
                DEVICE,
                experiment_config,
                pixel_threshold=pixel_threshold,
                image_threshold=image_threshold,
                criterion=criterion,
                collect_pr=save_pr,
            )

            plot_training_curves(artifacts["history"], plot_dir / "training_curves.png")
            plot_threshold_curve(checkpoint["val_stats"]["threshold_metrics"], plot_dir / "validation_threshold_curve.png")
            if save_pr:
                plot_pr_curves(test_result, plot_dir / "test_pr_curves.png")
            if save_visuals:
                plot_qualitative_grid(
                    model,
                    artifacts["loaders"]["test_dataset"],
                    DEVICE,
                    pixel_threshold,
                    test_result["per_image"],
                    plot_dir / "qualitative_failures.png",
                    num_examples=experiment_config["visualization_examples"],
                )

            per_image_df = pd.DataFrame(test_result["per_image"])
            per_image_df.to_csv(report_dir / "test_per_image_metrics.csv", index=False)

            robustness_df = None
            robustness_report = None
            if save_robustness:
                robustness_df, robustness_report = run_robustness_suite(
                    model,
                    experiment_config,
                    splits,
                    DEVICE,
                    pixel_threshold=pixel_threshold,
                    image_threshold=image_threshold,
                )
                robustness_df.to_csv(report_dir / "robustness_tampered_only.csv", index=False)

            mask_randomization_report = None
            if save_mask_randomization:
                mask_randomization_report = run_mask_randomization_test(
                    model,
                    artifacts["loaders"]["test_dataset"],
                    DEVICE,
                    experiment_config,
                    pixel_threshold=pixel_threshold,
                    image_threshold=image_threshold,
                    iterations=experiment_config["mask_randomization_iterations"],
                )

            summary = {
                "run_name": run_name,
                "selection_metric": experiment_config["primary_selection_metric"],
                "pixel_threshold": pixel_threshold,
                "image_threshold": image_threshold,
                "val_result": val_result,
                "test_result": test_result,
                "summary_row": summarize_result_row(run_name, test_result),
                "robustness_report": robustness_report,
                "mask_randomization_report": mask_randomization_report,
                "paths": artifacts["paths"],
            }
            save_json(report_dir / "evaluation_summary.json", summary)
            return summary


        def run_multi_seed_validation(base_config: dict, splits: dict, run_root: Path) -> dict:
            rows = []
            seed_outputs = []
            for seed in base_config["multi_seed_values"]:
                experiment_config = copy.deepcopy(base_config)
                experiment_config["seed"] = int(seed)
                experiment_config["run_multi_seed_validation"] = False
                experiment_config["run_architecture_comparison"] = False
                experiment_config["run_augmentation_ablation"] = False
                experiment_config["run_robustness_suite"] = False
                experiment_config["run_mask_randomization"] = False
                experiment_config["experiment_name"] = f"{base_config['experiment_name']}-seed{seed}"
                output = train_and_evaluate_once(
                    experiment_config,
                    splits,
                    ensure_dir(run_root / f"seed_{seed}"),
                    run_name=f"seed-{seed}",
                    save_visuals=False,
                    save_robustness=False,
                    save_mask_randomization=False,
                    save_pr=False,
                )
                row = output["summary_row"]
                row["seed"] = int(seed)
                rows.append(row)
                seed_outputs.append(output)

            summary = {
                "rows": rows,
                "mean_std": mean_std_summary(
                    rows,
                    [
                        "tampered_only_f1",
                        "tampered_only_iou",
                        "boundary_f1_mean",
                        "image_auc",
                        "image_f1",
                        "copymove_f1",
                        "splicing_f1",
                    ],
                ),
            }
            save_json(run_root / "multi_seed_summary.json", summary)
            pd.DataFrame(rows).to_csv(run_root / "multi_seed_summary.csv", index=False)
            return summary


        def run_architecture_comparison(base_config: dict, splits: dict, run_root: Path) -> dict:
            rows = []
            for architecture_name in ["unet", "deeplabv3plus"]:
                experiment_config = copy.deepcopy(base_config)
                experiment_config["architecture"] = architecture_name
                experiment_config["seed"] = 42
                experiment_config["run_multi_seed_validation"] = False
                experiment_config["run_architecture_comparison"] = False
                experiment_config["run_augmentation_ablation"] = False
                experiment_config["run_robustness_suite"] = False
                experiment_config["run_mask_randomization"] = False
                experiment_config["experiment_name"] = f"{base_config['experiment_name']}-{architecture_name}"
                output = train_and_evaluate_once(
                    experiment_config,
                    splits,
                    ensure_dir(run_root / architecture_name),
                    run_name=f"arch-{architecture_name}",
                    save_visuals=False,
                    save_robustness=False,
                    save_mask_randomization=False,
                    save_pr=False,
                )
                row = output["summary_row"]
                row["architecture"] = architecture_name
                rows.append(row)

            comparison = {"rows": rows}
            save_json(run_root / "architecture_comparison.json", comparison)
            pd.DataFrame(rows).to_csv(run_root / "architecture_comparison.csv", index=False)
            return comparison


        def run_augmentation_ablation(base_config: dict, splits: dict, run_root: Path) -> dict:
            variants = [
                (
                    "full_augmentation",
                    {
                        "aug_color_jitter": True,
                        "aug_compression": True,
                        "aug_gauss_noise": True,
                        "aug_gauss_blur": True,
                    },
                ),
                (
                    "geometric_only",
                    {
                        "aug_color_jitter": False,
                        "aug_compression": False,
                        "aug_gauss_noise": False,
                        "aug_gauss_blur": False,
                    },
                ),
            ]

            rows = []
            for variant_name, overrides in variants:
                experiment_config = copy.deepcopy(base_config)
                experiment_config.update(overrides)
                experiment_config["seed"] = 42
                experiment_config["run_multi_seed_validation"] = False
                experiment_config["run_architecture_comparison"] = False
                experiment_config["run_augmentation_ablation"] = False
                experiment_config["run_robustness_suite"] = True
                experiment_config["run_mask_randomization"] = False
                experiment_config["experiment_name"] = f"{base_config['experiment_name']}-{variant_name}"
                output = train_and_evaluate_once(
                    experiment_config,
                    splits,
                    ensure_dir(run_root / variant_name),
                    run_name=f"aug-{variant_name}",
                    save_visuals=False,
                    save_robustness=True,
                    save_mask_randomization=False,
                    save_pr=False,
                )
                row = output["summary_row"]
                row["variant"] = variant_name
                row["robustness_indicator_jpeg_qf50"] = maybe_float(
                    (output["robustness_report"] or {}).get("robustness_indicator_jpeg_qf50")
                )
                rows.append(row)

            comparison = {"rows": rows}
            save_json(run_root / "augmentation_ablation.json", comparison)
            pd.DataFrame(rows).to_csv(run_root / "augmentation_ablation.csv", index=False)
            return comparison
        """
    )


def narrative_cells(target: str) -> list[dict]:
    runtime_note = (
        "This generated notebook is the primary Google Colab submission artifact. "
        "The Kaggle notebook is a synchronized development convenience."
        if target == "colab"
        else "This generated notebook mirrors the Colab-first v9 pipeline for Kaggle development and comparison."
    )

    return [
        md(
            f"""
            # Tampered Image Detection & Localization (v9)

            **Variant:** `{target}`

            This notebook implements the Audit9-grounded v9 plan using the executed v8 run-01 as the engineering baseline.
            It keeps the good parts of the v8 line, fixes only Audit9-approved gaps, and keeps the narrative honest about what is improved versus what remains difficult.

            {runtime_note}
            """
        ),
        md(
            """
            ## Notebook Scope

            - CASIA v2.0 is treated as a **chosen baseline**, not a mandated dataset.
            - The notebook is an assignment-complete forensic baseline for analyst assistance, not a production system.
            - Copy-move improvement is **targeted but not guaranteed**. If it remains weak, that limitation must stay explicit.
            - Only Audit9-approved changes are implemented as new v9 behavior.
            """
        ),
        md(
            """
            ## 1. Setup, Environment, and Reproducibility

            This section installs dependencies, sets the Colab/Kaggle runtime contract, initializes paths, and prepares reproducible single-GPU-safe defaults.
            """
        ),
        md(
            """
            ## 2. Dataset Discovery and Leakage Control

            The data pipeline discovers image-mask pairs, documents the selected CASIA root, and performs a real pHash near-duplicate grouping step **before** train/val/test splitting.
            This replaces the weak path-only leakage checks criticized in the v8 run-01 audit.
            """
        ),
        md(
            """
            ## 3. Data Pipeline, Model, and Training

            The v9 model keeps the practical SMP baseline but adds the Audit9-approved dual-task classification head, RGB+ELA input, edge-aware loss, and per-forgery-type loss tracking.
            Validation, scheduler updates, early stopping, and primary checkpoint selection all use the same threshold-swept tampered-only Pixel-F1 objective.
            """
        ),
        md(
            """
            ## 4. Evaluation, Diagnostics, and Approved Experiments

            Evaluation leads with tampered-only localization metrics, forgery-type breakdowns, mask-size stratification, Boundary F1, PR curves, learned image-level metrics, mask randomization, and tampered-only robustness reporting.
            Audit9-approved heavier experiments are implemented but remain disabled by default.
            """
        ),
        md(
            """
            ## 5. Execution

            The default path runs one stable primary U-Net experiment with Audit9-approved core changes enabled.
            Multi-seed validation, DeepLabV3+ comparison, and augmentation ablation are available behind config flags and write explicit comparison reports when enabled.
            """
        ),
    ]


def execution_cell(target: str) -> dict:
    return code(
        f"""
        print("Starting Audit9-grounded v9 execution for {target}.")
        PRIMARY_RESULTS = None

        if CONFIG["run_primary_training"]:
            primary_run_dir = ensure_dir(OUTPUT_ROOT / "primary_run")
            PRIMARY_RESULTS = train_and_evaluate_once(
                CONFIG,
                SPLITS,
                primary_run_dir,
                run_name=f"{{CONFIG['experiment_name']}}-{target}-primary",
                save_visuals=True,
                save_robustness=CONFIG["run_robustness_suite"],
                save_mask_randomization=CONFIG["run_mask_randomization"],
                save_pr=True,
            )
            print("Primary run summary:")
            print(json.dumps(PRIMARY_RESULTS["summary_row"], indent=2))
            if PRIMARY_RESULTS["robustness_report"] is not None:
                print("Robustness indicator (clean - JPEG QF50 tampered-only F1):")
                print(PRIMARY_RESULTS["robustness_report"]["robustness_indicator_jpeg_qf50"])
            if PRIMARY_RESULTS["mask_randomization_report"] is not None:
                print("Mask randomization drop:")
                print(PRIMARY_RESULTS["mask_randomization_report"]["drop"])

        if CONFIG["run_multi_seed_validation"]:
            multi_seed_dir = ensure_dir(OUTPUT_ROOT / "multi_seed_validation")
            multi_seed_summary = run_multi_seed_validation(CONFIG, SPLITS, multi_seed_dir)
            print("Multi-seed summary:")
            print(json.dumps(multi_seed_summary["mean_std"], indent=2))

        if CONFIG["run_architecture_comparison"]:
            comparison_dir = ensure_dir(OUTPUT_ROOT / "architecture_comparison")
            architecture_summary = run_architecture_comparison(CONFIG, SPLITS, comparison_dir)
            print("Architecture comparison:")
            print(json.dumps(architecture_summary, indent=2))

        if CONFIG["run_augmentation_ablation"]:
            ablation_dir = ensure_dir(OUTPUT_ROOT / "augmentation_ablation")
            ablation_summary = run_augmentation_ablation(CONFIG, SPLITS, ablation_dir)
            print("Augmentation ablation summary:")
            print(json.dumps(ablation_summary, indent=2))

        print("Notebook execution scaffold complete.")
        print(f"Artifacts root: {{OUTPUT_ROOT}}")
        """
    )


def build_notebook(target: str) -> dict:
    intro, scope, setup_md, data_md, train_md, eval_md, execution_md = narrative_cells(target)
    cells = [
        intro,
        scope,
        setup_md,
        install_cell(),
        imports_cell(),
        config_cell(),
        runtime_cell(target),
        setup_cell(),
        data_md,
        dataset_discovery_cell(),
        phash_cell(),
        dataset_pipeline_cell(),
        train_md,
        model_cell(),
        loss_and_checkpoint_cell(),
        training_cell(),
        eval_md,
        evaluation_and_visuals_cell(),
        experiment_cell(),
        execution_md,
        execution_cell(target),
    ]

    if not cells:
        raise ValueError("Notebook assembly produced zero cells.")
    if not any(cell["cell_type"] == "markdown" for cell in cells):
        raise ValueError("Notebook assembly must include markdown cells.")
    if not any(cell["cell_type"] == "code" for cell in cells):
        raise ValueError("Notebook assembly must include code cells.")
    validate_code_cells(cells)
    return notebook(cells)


def write_notebook(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)


def main() -> None:
    kaggle_path = NOTEBOOK_DIR / "v9-tampered-image-detection-localization-kaggle.ipynb"
    colab_path = NOTEBOOK_DIR / "v9-tampered-image-detection-localization-colab.ipynb"

    kaggle_notebook = build_notebook("kaggle")
    colab_notebook = build_notebook("colab")

    write_notebook(kaggle_path, kaggle_notebook)
    write_notebook(colab_path, colab_notebook)

    print(f"Wrote {kaggle_path}")
    print(f"Wrote {colab_path}")


if __name__ == "__main__":
    main()
