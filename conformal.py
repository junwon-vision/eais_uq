"""
Conformal prediction: calibrate UQ thresholds with coverage guarantees.

Four calibration modes:
  image_id  — image-level, in-distribution
  image_cd  — image-level, correct detection
  traj_id   — trajectory-level, in-distribution
  traj_cd   — trajectory-level, correct detection
"""

import numpy as np

from uq_inference import batch_predict_uq, batch_predict_uq_laplace
from data_utils import _img_transform, decompress_image
from density import predict_density_episode

import torch


# ---------------------------------------------------------------------------
# Conformal quantile
# ---------------------------------------------------------------------------

def conformal_quantile(scores, alpha):
    """
    Compute the conformal quantile:  ceil((n+1)(1-alpha)) / n  quantile.
    Guarantees marginal coverage >= 1 - alpha.
    """
    n = len(scores)
    if n == 0:
        return float("nan")
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, q_level))


# ---------------------------------------------------------------------------
# Four calibration modes
# ---------------------------------------------------------------------------

def calibrate_image_id(results, alpha, uq_metric, alpha_traj=None):
    """Image-level, in-distribution: all timesteps treated as i.i.d."""
    scores = np.concatenate([r[uq_metric] for r in results])
    threshold = conformal_quantile(scores, alpha)
    return threshold, len(scores)


def calibrate_image_cd(results, alpha, uq_metric, alpha_traj=None):
    """Image-level, correct detection: only timesteps where prediction == GT."""
    scores = []
    for r in results:
        preds = (r["probs"] > 0.5).astype(float)
        correct = preds == r["labels"]
        scores.append(r[uq_metric][correct])
    scores = np.concatenate(scores)
    threshold = conformal_quantile(scores, alpha)
    return threshold, len(scores)


def calibrate_traj_id(results, alpha, uq_metric, alpha_traj=0.1):
    """Trajectory-level, in-distribution: 90th-percentile uncertainty per trajectory."""
    scores = np.array([np.percentile(r[uq_metric], 100 - alpha_traj * 100) for r in results])
    threshold = conformal_quantile(scores, alpha)
    return threshold, len(scores)


def calibrate_traj_cd(results, alpha, uq_metric, alpha_traj=0.1):
    """Trajectory-level, correct detection: 90th-pct UQ for correctly-classified trajs."""
    scores = []
    for r in results:
        traj_label = 1.0 if r["labels"].max() > 0.5 else 0.0
        traj_pred  = 1.0 if (r["probs"] > 0.5).mean() > 0.5 else 0.0
        if traj_label == traj_pred:
            scores.append(np.percentile(r[uq_metric], 100 - alpha_traj * 100))
    scores = np.array(scores)
    threshold = conformal_quantile(scores, alpha)
    return threshold, len(scores)


CALIBRATION_MODES = {
    "image_id": ("Image-level, in-distribution",   calibrate_image_id),
    "image_cd": ("Image-level, correct detection",  calibrate_image_cd),
    "traj_id":  ("Trajectory-level, in-distribution", calibrate_traj_id),
    "traj_cd":  ("Trajectory-level, correct detection", calibrate_traj_cd),
}


# ---------------------------------------------------------------------------
# End-to-end calibration
# ---------------------------------------------------------------------------

def _get_image(img):
    if isinstance(img, (bytes, np.bytes_)):
        return decompress_image(img)
    return img


def run_calibration(model, episodes, mode, device, alpha=0.1,
                    uq_metric="variances", laplace=None, batch_size=32,
                    logit_variance=False):
    """
    Run inference on calibration episodes and compute conformal thresholds
    for all four calibration modes.

    Returns:
        thresholds: dict  {cal_mode: {"threshold": float, "n_samples": int}}
        results:    list of per-episode dicts (for further analysis)
    """
    # --- inference ---
    results = []
    for ep_idx, ep in enumerate(episodes):
        labels = np.array(ep["failure"], dtype=np.float32)
        T = len(labels)
        all_probs, all_vars, all_ents = [], [], []

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
                        model, fronts, wrists, mode=mode, T=50,
                        logit_variance=logit_variance)
            all_probs.extend(probs)
            all_vars.extend(variances)
            all_ents.extend(entropies)

        results.append({
            "probs":     np.array(all_probs),
            "variances": np.array(all_vars),
            "entropies": np.array(all_ents),
            "labels":    labels,
        })

    # --- calibrate ---
    thresholds = {}
    for cal_key, (cal_desc, cal_fn) in CALIBRATION_MODES.items():
        threshold, n = cal_fn(results, alpha, uq_metric)
        thresholds[cal_key] = {"threshold": threshold, "n_samples": n,
                               "description": cal_desc}

    return thresholds, results


# ---------------------------------------------------------------------------
# Density-specific calibration (logpZO; higher = more OOD)
# ---------------------------------------------------------------------------

def run_calibration_density(density_model, encoder, episodes, device,
                            alpha=0.1, batch_size=32):
    """
    Run density-estimation calibration using logpZO scores.

    Calibrates image_id and traj_id modes only (no probs for cd modes).
    Returns:
        thresholds: dict {cal_key: {"threshold": float, "n_samples": int, "description": str}}
        results:    list of per-episode dicts {"logpZO": np.ndarray, "labels": np.ndarray}
    """
    results = []
    for ep in episodes:
        labels = np.array(ep["failure"], dtype=np.float32)
        logpZO = predict_density_episode(density_model, encoder, ep, device,
                                         batch_size=batch_size)
        results.append({"logpZO": logpZO, "labels": labels})

    thresholds = {}
    for cal_key, (cal_desc, cal_fn) in [
        ("image_id", CALIBRATION_MODES["image_id"]),
        ("traj_id",  CALIBRATION_MODES["traj_id"]),
    ]:
        threshold, n = cal_fn(results, alpha, "logpZO")
        thresholds[cal_key] = {"threshold": threshold, "n_samples": n,
                               "description": cal_desc}

    return thresholds, results


# ---------------------------------------------------------------------------
# Coverage evaluation
# ---------------------------------------------------------------------------

def evaluate_coverage(results, threshold, uq_metric="variances"):
    """
    Evaluate empirical coverage: fraction of samples with UQ < threshold.

    Returns dict with image-level and trajectory-level coverage stats.
    """
    all_uq = np.concatenate([r[uq_metric] for r in results])
    all_labels = np.concatenate([r["labels"] for r in results])
    all_probs = np.concatenate([r["probs"] for r in results])

    image_coverage = float((all_uq <= threshold).mean())

    preds = (all_probs > 0.5).astype(float)
    correct_mask = preds == all_labels
    if correct_mask.sum() > 0:
        correct_coverage = float((all_uq[correct_mask] <= threshold).mean())
    else:
        correct_coverage = float("nan")

    # Trajectory-level
    traj_max_uq = np.array([r[uq_metric].max() for r in results])
    traj_coverage = float((traj_max_uq <= threshold).mean())

    return {
        "image_coverage": image_coverage,
        "correct_detection_coverage": correct_coverage,
        "traj_coverage": traj_coverage,
        "n_images": len(all_uq),
        "n_trajs": len(results),
        "threshold": threshold,
    }
