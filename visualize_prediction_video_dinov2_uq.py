#!/usr/bin/env python3
"""
Generate UQ-only videos for DINOv2-based models.

Shows uncertainty graph only (no failure probability graph).
Failure indication via colored border: green (OK) / red (FAILURE).
R and B channels are inverted when UQ exceeds the threshold.

Usage:
    python visualize_prediction_video_dinov2_uq.py \
        --checkpoint checkpoints/failure_classifier_dinov2/ensemble/checkpoint.pt --mode ensemble \
        --calibration calibration.json --all
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eai_safety_hw.train_failure_classifier_dinov2 import (
    DATA_DIR,
    FailureClassifierDINOv2Ensemble,
    LaplaceWrapper,
    _img_transform,
    batch_predict_uq,
    batch_predict_uq_laplace,
    decompress_image,
)

OUTPUT_DIR = Path("visualizations/prediction_videos_dinov2_uq")
EVAL_FILE = "failure_eval.pkl"

MODE_UQ_METRIC = {
    "baseline": ("entropies", "Entropy"),
    "mc_dropout": ("variances", "Variance"),
    "ensemble": ("variances", "Variance"),
    "combined": ("variances", "Variance"),
    "laplace": ("variances", "Variance"),
}

MODE_CAL_METRIC = {
    "baseline": "entropy",
    "mc_dropout": "variance",
    "ensemble": "variance",
    "combined": "variance",
    "laplace": "variance",
}

BORDER_W = 8


def set_dropout_rate(model, rate):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = rate


def load_model(checkpoint_path, device, dropout_rate_override=None, mode=None):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ensemble_size = ckpt["ensemble_size"]
    train_dropout = ckpt.get("args", {}).get("dropout_rate", 0.2)
    freeze_backbone = not ckpt.get("args", {}).get("finetune_backbone", False)
    model = FailureClassifierDINOv2Ensemble(
        ensemble_size=ensemble_size,
        dropout_rate=train_dropout,
        freeze_backbone=freeze_backbone,
    )
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)

    eval_dropout = dropout_rate_override if dropout_rate_override is not None else train_dropout
    if dropout_rate_override is not None and dropout_rate_override != train_dropout:
        set_dropout_rate(model, dropout_rate_override)
        print(f"  Dropout rate overridden: {train_dropout} -> {eval_dropout}")

    laplace = None
    if mode == "laplace" and "laplace_state" in ckpt:
        model_single = model.members[0]
        laplace = LaplaceWrapper(
            model_single,
            prior_precision=ckpt["laplace_state"]["prior_precision"],
        )
        laplace.posterior_cov = ckpt["laplace_state"]["posterior_cov"].to(device)
        laplace.fitted = True
        print(f"  Laplace state loaded (prior_precision={laplace.prior_precision})")

    model_name = f"DINOv2 K={ensemble_size}, drop={eval_dropout}"
    print(f"Loaded model: {model_name}, from {checkpoint_path}")
    return model, model_name, laplace


def run_inference_episode(model, ep, mode, device, laplace=None, batch_size=32,
                         logit_variance=False):
    T = len(ep["failure"])
    all_probs, all_vars, all_entropies = [], [], []

    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        fronts, wrists = [], []
        for t in range(start, end):
            f_img = decompress_image(ep["front_cam"][t])
            w_img = decompress_image(ep["wrist_cam"][t])
            fronts.append(_img_transform(f_img))
            wrists.append(_img_transform(w_img))
        fronts = torch.stack(fronts).to(device)
        wrists = torch.stack(wrists).to(device)

        if mode == "laplace" and laplace is not None:
            probs, variances, entropies = batch_predict_uq_laplace(laplace, fronts, wrists)
        else:
            with torch.no_grad():
                probs, variances, entropies = batch_predict_uq(
                    model, fronts, wrists, mode=mode, T=20,
                    logit_variance=logit_variance,
                )
        all_probs.extend(probs)
        all_vars.extend(variances)
        all_entropies.extend(entropies)

    return np.array(all_probs), np.array(all_vars), np.array(all_entropies)


def render_graph(t, T, uq_scores, mode, graph_width, graph_height,
                 uq_label="UQ", threshold=None, uq_ymax=None):
    """Render single-row UQ graph up to timestep t as an RGB image."""
    dpi = 100
    fig_w = graph_width / dpi
    fig_h = graph_height / dpi
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=dpi)

    x_so_far = np.arange(t + 1)
    uq_vals = uq_scores[:t + 1]

    if threshold is not None:
        ax.axhline(y=threshold, color="gray", linestyle=":", linewidth=1.2, alpha=0.8,
                   label=f"threshold={threshold:.4f}")
        for i in range(len(uq_vals) - 1):
            seg_x = [i, i + 1]
            seg_y = [uq_vals[i], uq_vals[i + 1]]
            if uq_vals[i] > threshold or uq_vals[i + 1] > threshold:
                ax.plot(seg_x, seg_y, color="red", linewidth=1.8)
            else:
                ax.plot(seg_x, seg_y, color="purple", linewidth=1.5)
        if len(uq_vals) == 1:
            color = "red" if uq_vals[0] > threshold else "purple"
            ax.plot([0], [uq_vals[0]], color=color, marker="o", markersize=3)
        ax.legend(fontsize=7, loc="upper right")
    else:
        ax.plot(x_so_far, uq_vals, color="purple", linewidth=1.5)

    ax.fill_between(x_so_far, 0, uq_vals, color="purple", alpha=0.1)
    ax.set_xlim(0, T)
    ax.set_ylim(0, uq_ymax)
    ax.set_ylabel(uq_label, fontsize=8)
    ax.set_xlabel("Timestep", fontsize=8)
    ax.set_title(f"Predictive {uq_label} (UQ) — {mode}", fontsize=9)
    ax.tick_params(labelsize=7)

    fig.tight_layout(pad=0.5)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(int(fig.get_figheight() * dpi), int(fig.get_figwidth() * dpi), 3)
    plt.close(fig)
    return buf


def render_frame(ep, t, probs, uq_scores, uq_label, mode, T,
                 threshold=None, uq_ymax=None):
    """Compose a full video frame: cameras with colored border on top, UQ graph below."""
    front = decompress_image(ep["front_cam"][t])
    wrist = decompress_image(ep["wrist_cam"][t])

    # Invert R and B channels if UQ above threshold
    if threshold is not None and uq_scores[t] > threshold:
        front = front[:, :, ::-1].copy()
        wrist = wrist[:, :, ::-1].copy()

    sep = np.zeros((256, 4, 3), dtype=np.uint8)
    cam_row = np.concatenate([front, sep, wrist], axis=1)

    # Add colored border: green if P(fail) < 0.5, red if > 0.5
    prob_now = probs[t]
    border_color = np.array([255, 0, 0], dtype=np.uint8) if prob_now > 0.5 \
        else np.array([0, 255, 0], dtype=np.uint8)
    cam_row[:BORDER_W, :] = border_color
    cam_row[-BORDER_W:, :] = border_color
    cam_row[:, :BORDER_W] = border_color
    cam_row[:, -BORDER_W:] = border_color

    cam_w = cam_row.shape[1]

    # Header
    header_h = 32
    header = np.zeros((header_h, cam_w, 3), dtype=np.uint8) + 30
    header_bgr = header[:, :, ::-1].copy()

    uq_now = uq_scores[t]
    pred = "FAILURE" if prob_now > 0.5 else "OK"
    pred_color = (60, 60, 255) if prob_now > 0.5 else (60, 200, 60)

    cv2.putText(header_bgr, f"t={t}/{T}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(header_bgr, pred, (120, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, pred_color, 2)
    cv2.putText(header_bgr, f"P={prob_now:.3f}  {uq_label}={uq_now:.4f}",
                (230, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(header_bgr, "FRONT", (50, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
    cv2.putText(header_bgr, "WRIST", (cam_w // 2 + 50, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
    header = header_bgr[:, :, ::-1].copy()

    # UQ-only graph
    graph_h = 200
    graph = render_graph(t, T, uq_scores, mode,
                         graph_width=cam_w, graph_height=graph_h,
                         uq_label=uq_label, threshold=threshold,
                         uq_ymax=uq_ymax)
    if graph.shape[1] != cam_w:
        graph = cv2.resize(graph, (cam_w, graph_h))

    frame = np.concatenate([header, cam_row, graph], axis=0)
    return frame


def render_episode(ep, ep_idx, probs, uq_scores, uq_label, mode, fps,
                   threshold=None, uq_ymax=None, output_dir=None):
    """Render and save video for a single episode."""
    T = len(probs)
    frames = []
    for t in range(T):
        frame = render_frame(ep, t, probs, uq_scores, uq_label, mode, T,
                             threshold=threshold, uq_ymax=uq_ymax)
        frames.append(frame)

    save_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"ep{ep_idx:04d}_{mode}_uq_T{T}.mp4"
    imageio.mimwrite(str(out_path), frames, fps=fps, codec="libx264",
                     macro_block_size=1,
                     output_params=["-pix_fmt", "yuv420p"])
    print(f"  Saved: {out_path}")
    return out_path


def load_calibration_threshold(cal_path, mode, cal_mode="image_id"):
    with open(cal_path, "r") as f:
        cal = json.load(f)
    metric_key = MODE_CAL_METRIC.get(mode, "variance")
    thresholds = cal.get("thresholds", {})
    if metric_key in thresholds and cal_mode in thresholds[metric_key]:
        return thresholds[metric_key][cal_mode]["threshold"]
    print(f"  Warning: no threshold found for metric={metric_key}, cal_mode={cal_mode}")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate UQ-only prediction videos (DINOv2) — border color = failure")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mode", type=str, default="ensemble",
                        choices=["baseline", "mc_dropout", "ensemble", "combined", "laplace"])
    parser.add_argument("--pkl", type=str, default=EVAL_FILE)
    parser.add_argument("--episode", type=int, nargs="*", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--dropout_rate", type=float, default=None)
    parser.add_argument("--calibration", type=str, default=None)
    parser.add_argument("--cal_mode", type=str, default="image_id",
                        choices=["image_id", "image_cd", "traj_id", "traj_cd"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--logit_variance", action="store_true",
                        help="Compute variance in logit space instead of probability space")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, model_name, laplace = load_model(
        args.checkpoint, device,
        dropout_rate_override=args.dropout_rate, mode=args.mode,
    )

    threshold = None
    if args.calibration:
        threshold = load_calibration_threshold(args.calibration, args.mode, args.cal_mode)
        if threshold is not None:
            print(f"Calibration threshold ({args.cal_mode}): {threshold:.6f}")

    pkl_path = DATA_DIR / args.pkl
    print(f"Loading data: {pkl_path}")
    with open(pkl_path, "rb") as f:
        episodes = pickle.load(f)
    print(f"Loaded {len(episodes)} episodes")

    if args.all:
        indices = list(range(len(episodes)))
    elif args.episode is not None:
        indices = args.episode
    else:
        indices = [0]

    uq_key, uq_label = MODE_UQ_METRIC.get(args.mode, ("variances", "Variance"))

    print(f"\nRunning inference on {len(indices)} episode(s)...")
    ep_results = {}
    for idx in indices:
        if idx >= len(episodes):
            print(f"Skipping episode {idx} (out of range)")
            continue
        ep = episodes[idx]
        T = len(ep["failure"])
        print(f"  Episode {idx}: T={T}", end="\r")
        probs, variances, entropies = run_inference_episode(
            model, ep, args.mode, device, laplace=laplace,
            logit_variance=args.logit_variance)
        uq_scores = entropies if uq_key == "entropies" else variances
        ep_results[idx] = (probs, uq_scores)
    print()

    global_uq_max = max(uq.max() for _, uq in ep_results.values())
    uq_ymax = max(global_uq_max * 1.2, 0.01)
    if threshold is not None:
        uq_ymax = max(uq_ymax, threshold * 1.5)
    print(f"UQ y-axis fixed to [0, {uq_ymax:.4f}] (metric={uq_label})")

    print(f"Rendering {len(ep_results)} video(s)...")
    for idx, (probs, uq_scores) in ep_results.items():
        render_episode(episodes[idx], idx, probs, uq_scores, uq_label,
                       args.mode, args.fps, threshold=threshold,
                       uq_ymax=uq_ymax, output_dir=args.output_dir)

    out = args.output_dir or str(OUTPUT_DIR)
    print(f"\nDone! Videos saved to {out}/")


if __name__ == "__main__":
    main()
