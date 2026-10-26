"""
Tests for DVC pipeline configuration and stage definitions.
"""
from pathlib import Path
import yaml
import pytest


def test_dvc_pipeline_configuration():
    """Verify that dvc.yaml contains all required stages, dependencies, and outputs."""
    dvc_path = Path("dvc.yaml")
    assert dvc_path.exists(), "dvc.yaml does not exist"

    with open(dvc_path, "r", encoding="utf-8") as f:
        dvc_config = yaml.safe_load(f)

    assert "stages" in dvc_config
    stages = dvc_config["stages"]
    assert "verify_data" in stages
    assert "train" in stages
    assert "evaluate" in stages

    # Check stage dependencies
    assert "data/pneumoniamnist.npz" in stages["verify_data"]["deps"]
    assert "artifacts/best_model.pt" in stages["evaluate"]["deps"]
