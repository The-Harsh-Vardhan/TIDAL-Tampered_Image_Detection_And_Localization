"""Model architecture and checkpoint compatibility tests."""

from pathlib import Path

import torch

from backend.inference.model_architecture import CBAMBlock, build_vrp301_model

MODEL_PATH = Path("models") / "inference" / "vR.P.30.1" / "best_model.pt"


def test_vrp301_model_injects_cbam_blocks():
    model = build_vrp301_model()
    assert len(model.decoder.blocks) == 5
    assert all(isinstance(block.attention2, CBAMBlock) for block in model.decoder.blocks)


def test_vrp301_checkpoint_loads_without_shape_mismatch():
    model = build_vrp301_model()
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state_dict, strict=True)
