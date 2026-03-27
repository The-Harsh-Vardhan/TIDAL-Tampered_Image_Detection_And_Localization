"""Validate DVC pipeline structure."""
from pathlib import Path
import yaml, pytest
DVC = Path(__file__).parent.parent / "dvc_pipeline"

def test_dvc_yaml_valid():
    with open(DVC/"dvc.yaml") as f: c=yaml.safe_load(f)
    assert set(c["stages"].keys()) == {"preprocess","train","evaluate","visualize"}

def test_params_yaml_valid():
    with open(DVC/"params.yaml") as f: p=yaml.safe_load(f)
    assert p["seed"]==42 and p["preprocessing"]["ela_qualities"]==[75,85,95]

@pytest.mark.parametrize("f",["src/utils.py","src/dataset.py","src/models.py","src/train.py","src/evaluate.py","src/visualize.py","src/preprocess.py"])
def test_source_exists(f): assert (DVC/f).exists()
