"""
Utility functions for configuration, device management, and reproducibility.
"""
from pathlib import Path
import random
import os
import yaml
import numpy as np
import torch


def find_project_root() -> Path:
    """Find the root directory of the project."""
    current = Path(__file__).resolve().parent
    while current.parent != current:
        if (current / "configs").is_dir() or (current / "requirements.txt").is_file():
            return current
        current = current.parent
    return Path.cwd()


def load_config(config_path: str | Path | None = None) -> dict:
    """Load configuration YAML file."""
    root = find_project_root()
    if config_path is None:
        primary = root / "configs" / "config.yaml"
    else:
        primary = Path(config_path)
        if not primary.is_absolute():
            primary = root / primary

    if not primary.exists():
        raise FileNotFoundError(f"Configuration file not found at: {primary}")

    with open(primary, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_device(preference: str = "auto") -> torch.device:
    """
    Select computing device (CUDA if available and requested, otherwise CPU).
    Logs the selection reason clearly.
    """
    pref = (preference or "auto").lower()
    if pref in ["cuda", "gpu"]:
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[Device] CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    elif pref == "cpu":
        return torch.device("cpu")
    else:  # 'auto'
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    """Set random seed across random, numpy, and torch for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_data_path(config: dict) -> Path:
    """
    Locates the dataset file using paths defined in config,
    checking data_path, fallback_path, and common relative locations.
    """
    root = find_project_root()
    candidates = [
        root / config.get("data", {}).get("data_path", "data/pneumoniamnist.npz"),
        root / config.get("data", {}).get("fallback_path", "pneumoniamnist.npz"),
        root / "pneumoniamnist.npz",
        root / "data" / "pneumoniamnist.npz",
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return path

    raise FileNotFoundError(
        f"PneumoniaMNIST dataset file not found. Looked in: {[str(c) for c in candidates]}. "
        "Please place pneumoniamnist.npz in the data/ directory or repository root."
    )
