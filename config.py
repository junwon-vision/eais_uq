"""
Central configuration for the UQ homework.

All paths, constants, and device settings live here so that every other module
in this package can simply ``from config import *``.
"""

from pathlib import Path
import torch

# ---------------------------------------------------------------------------
# Paths  (all data lives under  final_hw/data/)
# ---------------------------------------------------------------------------
_DATA_ROOT = Path(__file__).resolve().parent / "data"

# Dataset pickle files (episodes)
DATA_DIR = _DATA_ROOT

# Model checkpoints
CHECKPOINT_DIR = _DATA_ROOT / "checkpoints"

# OOD trajectory datasets
OOD_DIR         = _DATA_ROOT / "ood_data" / "failure"   # real OOD failure trajs
OOD_SUCCESS_DIR = _DATA_ROOT / "ood_data" / "ood" / "success"
OOD_FAILURE_DIR = _DATA_ROOT / "ood_data" / "ood" / "failure"

# Pre-trained classifier checkpoints (DINOv2 backbone).
CLASSIFIER_CHECKPOINTS = {
    "baseline":   CHECKPOINT_DIR / "failure_classifier_dinov2" / "baseline"   / "checkpoint.pt",
    "mc_dropout": CHECKPOINT_DIR / "failure_classifier_dinov2" / "mc_dropout" / "checkpoint.pt",
    "ensemble":   CHECKPOINT_DIR / "failure_classifier_dinov2" / "ensemble"   / "checkpoint.pt",
}

# Density estimator checkpoint (flow matching on DINOv2 features).
DENSITY_CHECKPOINT = CHECKPOINT_DIR / "density_dinov2" / "checkpoint.pt"

# ---------------------------------------------------------------------------
# Image pre-processing constants (DINOv2 / ImageNet)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
