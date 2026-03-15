"""
UQ inference routines: MC Dropout, Ensemble, Laplace, and combined.
"""

import numpy as np
import torch
import torch.nn as nn

from data_utils import _img_transform, decompress_image
from models import FailureClassifierDINOv2Ensemble


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def compute_entropy(p):
    """Binary entropy H(p) for probability p (scalar or array)."""
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return -p * np.log(p) - (1 - p) * np.log(1 - p)


def enable_mc_dropout(model):
    """Set model to eval but keep Dropout layers in train mode."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


# ---------------------------------------------------------------------------
# Core MC-Dropout forward pass
# ---------------------------------------------------------------------------

def mc_dropout_predict(model, front, wrist, T=50):
    """Run T stochastic forward passes. Returns (T, B) logits tensor."""
    enable_mc_dropout(model)
    logits = []
    with torch.no_grad():
        for _ in range(T):
            logits.append(model(front, wrist))
    return torch.stack(logits, dim=0)


# ---------------------------------------------------------------------------
# Batch prediction (already-prepared tensors)
# ---------------------------------------------------------------------------

def batch_predict_uq(model, front, wrist, mode="mc_dropout", T=50,
                     logit_variance=False):
    """
    Batch UQ prediction on already-prepared tensors (B, C, H, W).

    Args:
        model: FailureClassifierDINOv2 or FailureClassifierDINOv2Ensemble
        front, wrist: (B, 3, 224, 224) tensors on device
        mode: "baseline" | "mc_dropout" | "ensemble" | "combined"
        T: number of MC samples
        logit_variance: if True compute variance in logit space

    Returns:
        mean_probs: (B,) numpy array
        variances:  (B,) numpy array
        entropies:  (B,) numpy array
    """
    if mode == "baseline":
        m = model.members[0] if isinstance(model, FailureClassifierDINOv2Ensemble) else model
        m.eval()
        with torch.no_grad():
            logits = m(front, wrist)
        all_logits = logits.unsqueeze(0)

    elif mode == "mc_dropout":
        m = model.members[0] if isinstance(model, FailureClassifierDINOv2Ensemble) else model
        all_logits = mc_dropout_predict(m, front, wrist, T=T)

    elif mode == "ensemble":
        model.eval()
        with torch.no_grad():
            all_logits = model(front, wrist)  # (K, B)

    elif mode == "combined":
        logit_list = []
        for member in model.members:
            member_logits = mc_dropout_predict(member, front, wrist, T=T)
            logit_list.append(member_logits)
        all_logits = torch.cat(logit_list, dim=0)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    logits_np = all_logits.cpu().numpy()
    probs_np = 1.0 / (1.0 + np.exp(-logits_np))
    mean_probs = probs_np.mean(axis=0)
    variances = logits_np.var(axis=0) if logit_variance else probs_np.var(axis=0)
    entropies = compute_entropy(mean_probs)
    return mean_probs, variances, entropies


def batch_predict_uq_laplace(laplace_wrapper, front, wrist):
    """Thin wrapper around LaplaceWrapper.predict()."""
    return laplace_wrapper.predict(front, wrist)


# ---------------------------------------------------------------------------
# Single-image prediction (from raw bytes)
# ---------------------------------------------------------------------------

def predict_single(model, front_bytes, wrist_bytes, mode="mc_dropout",
                   device="cuda", T=50, logit_variance=False):
    """
    Predict on a single (front, wrist) image pair given as JPEG bytes.

    Returns dict: {"prob": float, "variance": float, "entropy": float}
    """
    front_img = decompress_image(front_bytes)
    wrist_img = decompress_image(wrist_bytes)
    front = _img_transform(front_img).unsqueeze(0).to(device)
    wrist = _img_transform(wrist_img).unsqueeze(0).to(device)

    probs, variances, entropies = batch_predict_uq(
        model, front, wrist, mode=mode, T=T, logit_variance=logit_variance)
    return {
        "prob": float(probs[0]),
        "variance": float(variances[0]),
        "entropy": float(entropies[0]),
    }


# ---------------------------------------------------------------------------
# Full-episode inference
# ---------------------------------------------------------------------------

def _get_image(img):
    """Handle both JPEG bytes and raw numpy arrays."""
    if isinstance(img, (bytes, np.bytes_)):
        return decompress_image(img)
    return img


def run_inference_episode(model, ep, mode, device, laplace=None,
                          batch_size=32, logit_variance=False):
    """
    Run UQ inference on every timestep of an episode.

    Returns:
        probs:     (T,) numpy array
        variances: (T,) numpy array
        entropies: (T,) numpy array
    """
    T = len(ep["failure"])
    all_probs, all_vars, all_entropies = [], [], []

    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        fronts, wrists = [], []
        for t in range(start, end):
            fronts.append(_img_transform(_get_image(ep["front_cam"][t])))
            wrists.append(_img_transform(_get_image(ep["wrist_cam"][t])))
        fronts = torch.stack(fronts).to(device)
        wrists = torch.stack(wrists).to(device)

        if mode == "laplace" and laplace is not None:
            probs, variances, entropies = batch_predict_uq_laplace(
                laplace, fronts, wrists)
        else:
            with torch.no_grad():
                probs, variances, entropies = batch_predict_uq(
                    model, fronts, wrists, mode=mode, T=20,
                    logit_variance=logit_variance)
        all_probs.extend(probs)
        all_vars.extend(variances)
        all_entropies.extend(entropies)

    return np.array(all_probs), np.array(all_vars), np.array(all_entropies)
