"""
tests/test_dvc_pipeline.py
============================
Validate DVC pipeline structure and configuration.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml


DVC_PIPELINE_DIR = Path(__file__).parent.parent / "dvc_pipeline"


class TestDvcYaml:
    """Tests for dvc.yaml structure validation."""

    def test_dvc_yaml_exists(self):
        assert (DVC_PIPELINE_DIR / "dvc.yaml").exists()

    def test_dvc_yaml_parses(self):
        with open(DVC_PIPELINE_DIR / "dvc.yaml") as f:
            config = yaml.safe_load(f)
        assert "stages" in config

    def test_has_required_stages(self):
        with open(DVC_PIPELINE_DIR / "dvc.yaml") as f:
            config = yaml.safe_load(f)
        stages = config["stages"]
        required = ["preprocess", "train", "evaluate", "visualize"]
        for stage in required:
            assert stage in stages, f"Missing stage: {stage}"

    def test_stages_have_cmd(self):
        with open(DVC_PIPELINE_DIR / "dvc.yaml") as f:
            config = yaml.safe_load(f)
        for name, stage in config["stages"].items():
            assert "cmd" in stage, f"Stage {name} missing cmd"

    def test_stages_have_deps(self):
        with open(DVC_PIPELINE_DIR / "dvc.yaml") as f:
            config = yaml.safe_load(f)
        for name, stage in config["stages"].items():
            assert "deps" in stage, f"Stage {name} missing deps"

    def test_train_depends_on_preprocess_output(self):
        with open(DVC_PIPELINE_DIR / "dvc.yaml") as f:
            config = yaml.safe_load(f)
        train_deps = config["stages"]["train"]["deps"]
        assert "artifacts/ela_statistics.json" in train_deps


class TestParamsYaml:
    """Tests for params.yaml structure."""

    def test_params_yaml_exists(self):
        assert (DVC_PIPELINE_DIR / "params.yaml").exists()

    def test_params_yaml_parses(self):
        with open(DVC_PIPELINE_DIR / "params.yaml") as f:
            params = yaml.safe_load(f)
        assert isinstance(params, dict)

    def test_has_required_sections(self):
        with open(DVC_PIPELINE_DIR / "params.yaml") as f:
            params = yaml.safe_load(f)
        required = ["seed", "data", "preprocessing", "model", "training", "evaluation"]
        for section in required:
            assert section in params, f"Missing section: {section}"

    def test_seed_is_42(self):
        with open(DVC_PIPELINE_DIR / "params.yaml") as f:
            params = yaml.safe_load(f)
        assert params["seed"] == 42

    def test_ela_qualities(self):
        with open(DVC_PIPELINE_DIR / "params.yaml") as f:
            params = yaml.safe_load(f)
        qualities = params["preprocessing"]["ela_qualities"]
        assert qualities == [75, 85, 95]

    def test_in_channels_is_9(self):
        with open(DVC_PIPELINE_DIR / "params.yaml") as f:
            params = yaml.safe_load(f)
        assert params["preprocessing"]["in_channels"] == 9
        assert params["model"]["in_channels"] == 9


class TestSourceFiles:
    """Verify all DVC source files exist."""

    @pytest.mark.parametrize("filename", [
        "src/utils.py",
        "src/dataset.py",
        "src/models.py",
        "src/train.py",
        "src/evaluate.py",
        "src/visualize.py",
        "src/preprocess.py",
    ])
    def test_source_file_exists(self, filename: str):
        assert (DVC_PIPELINE_DIR / filename).exists(), f"Missing: {filename}"
