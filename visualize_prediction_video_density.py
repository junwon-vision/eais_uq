#!/usr/bin/env python3
"""
Generate videos with failure prediction and density-based UQ for DINOv2 models.

Combines the DINOv2 failure classifier (for P(fail)) with the flow matching
density estimator (logpZO as UQ score). Higher logpZO = lower density = more OOD.

Usage:
    python eai_safety_hw/visualize_prediction_video_density.py \
        --classifier_checkpoint checkpoints/failure_classifier_dinov2/ensemble/checkpoint.pt \
        --density_checkpoint checkpoints/density_dinov2/checkpoint.pt \
        --mode ensemble --all

    # Single episode
    python eai_safety_hw/visualize_prediction_video_density.py \
        --classifier_checkpoint checkpoints/failure_classifier_dinov2/baseline/checkpoint.pt \
        --density_checkpoint checkpoints/density_dinov2/checkpoint.pt \
        --mode baseline --episode 0 1 2
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
from eai_safety_hw.train_density_dinov2 import (
    load_density_model,
    predict_density_episode,
)

OUTPUT_DIR = Path("visualizations/prediction_videos_density")
OOD_DIR = Path("eai_safety_hw/ood_data/failure")
EVAL_FILE = "failure_eval.pkl"

MODE_UQ_METRIC = {
    "baseline": ("entropies", "Entropy"),
    "mc_dropout": ("variances", "Variance"),
    "ensemble": ("variances", "Variance"),
    "combined": ("variances", "Variance"),
    "laplace": ("variances", "Variance"),
}


def _get_image(img):
    """Convert image to RGB numpy array. Handles both JPEG bytes and raw numpy."""
    if isinstance(img, (bytes, np.bytes_)):
        return decompress_image(img)
    return img


# ---------------------------------------------------------------------------
# OOD data loading
# ---------------------------------------------------------------------------

def load_ood_trajectory(pkl_path):
    """Load a single OOD trajectory pkl file."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    traj = data[0]

    front_imgs, wrist_imgs, labels = [], [], []
    for info, _action, label in traj:
        front_imgs.append(info["cam_rs"][0])
        wrist_imgs.append(info["cam_zed_right"][0])
        labels.append(label)

    return {
        "front_cam": front_imgs,
        "wrist_cam": wrist_imgs,
        "failure": labels,
        "source": Path(pkl_path).stem,
    }


def load_ood_all(ood_dir=OOD_DIR):
    """Load all OOD trajectory pkl files from a directory."""
    ood_dir = Path(ood_dir)
    pkl_files = sorted(ood_dir.glob("traj_*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No traj_*.pkl found in {ood_dir}")
    return [load_ood_trajectory(p) for p in pkl_files]


def load_classifier(checkpoint_path, device, mode=None):
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

    laplace = None
    if mode == "laplace" and "laplace_state" in ckpt:
        model_single = model.members[0]
        laplace = LaplaceWrapper(
            model_single,
            prior_precision=ckpt["laplace_state"]["prior_precision"],
        )
        laplace.posterior_cov = ckpt["laplace_state"]["posterior_cov"].to(device)
        laplace.fitted = True

    print(f"Loaded DINOv2 classifier: K={ensemble_size}, dropout={train_dropout}")
    return model, laplace


def run_classifier_episode(model, ep, mode, device, laplace=None, batch_size=32):
    """Run failure classifier on an episode. Returns probs, variances, entropies."""
    T = len(ep["failure"])
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
            probs, variances, entropies = batch_predict_uq_laplace(laplace, fronts, wrists)
        else:
            with torch.no_grad():
                probs, variances, entropies = batch_predict_uq(
                    model, fronts, wrists, mode=mode, T=20)
        all_probs.extend(probs)
        all_vars.extend(variances)
        all_ents.extend(entropies)

    return np.array(all_probs), np.array(all_vars), np.array(all_ents)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_graph(t, T, probs, logpZO_scores, classifier_uq, mode,
                 graph_width, graph_height, uq_label, logpZO_ymax, uq_ymax,
                 threshold=None):
    """3-row graph: P(fail), classifier UQ, logpZO density."""
    dpi = 100
    fig_w = graph_width / dpi
    fig_h = graph_height / dpi
    fig, axes = plt.subplots(3, 1, figsize=(fig_w, fig_h), dpi=dpi)

    x_so_far = np.arange(t + 1)

    # Row 1: Failure probability
    ax = axes[0]
    ax.plot(x_so_far, probs[:t + 1], color="black", linewidth=1.5)
    ax.axhline(y=0.5, color="orange", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.fill_between(x_so_far, probs[:t + 1], 0.5,
                     where=probs[:t + 1] > 0.5, color="red", alpha=0.25)
    ax.set_xlim(0, T)
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(fail)", fontsize=7)
    ax.set_title(f"Failure Probability — {mode}", fontsize=8)
    ax.tick_params(labelsize=6)

    # Row 2: Classifier UQ metric
    ax = axes[1]
    uq_vals = classifier_uq[:t + 1]
    if threshold is not None:
        ax.axhline(y=threshold, color="gray", linestyle=":", linewidth=1.0, alpha=0.8,
                   label=f"τ={threshold:.4f}")
        for i in range(len(uq_vals) - 1):
            seg_x = [i, i + 1]
            seg_y = [uq_vals[i], uq_vals[i + 1]]
            color = "red" if uq_vals[i] > threshold or uq_vals[i + 1] > threshold else "purple"
            ax.plot(seg_x, seg_y, color=color, linewidth=1.5)
        if len(uq_vals) == 1:
            color = "red" if uq_vals[0] > threshold else "purple"
            ax.plot([0], [uq_vals[0]], color=color, marker="o", markersize=3)
        ax.legend(fontsize=6, loc="upper right")
    else:
        ax.plot(x_so_far, uq_vals, color="purple", linewidth=1.5)
    ax.fill_between(x_so_far, 0, uq_vals, color="purple", alpha=0.1)
    ax.set_xlim(0, T)
    ax.set_ylim(0, uq_ymax)
    ax.set_ylabel(uq_label, fontsize=7)
    ax.set_title(f"Predictive {uq_label}", fontsize=8)
    ax.tick_params(labelsize=6)

    # Row 3: logpZO (density UQ)
    ax = axes[2]
    logp_vals = logpZO_scores[:t + 1]
    ax.plot(x_so_far, logp_vals, color="darkgreen", linewidth=1.5)
    ax.fill_between(x_so_far, 0, logp_vals, color="green", alpha=0.1)
    ax.set_xlim(0, T)
    ax.set_ylim(0, logpZO_ymax)
    ax.set_ylabel("logpZO", fontsize=7)
    ax.set_xlabel("Timestep", fontsize=7)
    ax.set_title("Density UQ (higher = more OOD)", fontsize=8)
    ax.tick_params(labelsize=6)

    fig.tight_layout(pad=0.3)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(int(fig.get_figheight() * dpi), int(fig.get_figwidth() * dpi), 3)
    plt.close(fig)
    return buf


def render_frame(ep, t, probs, logpZO_scores, classifier_uq, uq_label, mode, T,
                 logpZO_ymax, uq_ymax, threshold=None):
    front = _get_image(ep["front_cam"][t])
    wrist = _get_image(ep["wrist_cam"][t])
    sep = np.zeros((256, 4, 3), dtype=np.uint8)
    cam_row = np.concatenate([front, sep, wrist], axis=1)
    cam_w = cam_row.shape[1]

    header_h = 32
    header = np.zeros((header_h, cam_w, 3), dtype=np.uint8) + 30
    header_bgr = header[:, :, ::-1].copy()

    prob_now = probs[t]
    logp_now = logpZO_scores[t]
    pred = "FAILURE" if prob_now > 0.5 else "OK"
    pred_color = (60, 60, 255) if prob_now > 0.5 else (60, 200, 60)

    cv2.putText(header_bgr, f"t={t}/{T}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(header_bgr, pred, (120, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, pred_color, 2)
    cv2.putText(header_bgr, f"P={prob_now:.3f}  logpZO={logp_now:.1f}",
                (230, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    header = header_bgr[:, :, ::-1].copy()

    graph_h = 400
    graph = render_graph(t, T, probs, logpZO_scores, classifier_uq, mode,
                         graph_width=cam_w, graph_height=graph_h,
                         uq_label=uq_label, logpZO_ymax=logpZO_ymax,
                         uq_ymax=uq_ymax, threshold=threshold)
    if graph.shape[1] != cam_w:
        graph = cv2.resize(graph, (cam_w, graph_h))

    frame = np.concatenate([header, cam_row, graph], axis=0)
    return frame


def render_episode(ep, ep_idx, probs, logpZO_scores, classifier_uq, uq_label,
                   mode, fps, logpZO_ymax, uq_ymax, threshold=None, output_dir=None,
                   name_prefix=None):
    T = len(probs)
    frames = []
    for t in range(T):
        frame = render_frame(ep, t, probs, logpZO_scores, classifier_uq,
                             uq_label, mode, T, logpZO_ymax, uq_ymax, threshold)
        frames.append(frame)

    save_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    prefix = name_prefix or f"ep{ep_idx:04d}"
    out_path = save_dir / f"{prefix}_{mode}_density_T{T}.mp4"
    imageio.mimwrite(str(out_path), frames, fps=fps, codec="libx264",
                     macro_block_size=1,
                     output_params=["-pix_fmt", "yuv420p"])
    print(f"  Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prediction videos with classifier + density UQ (DINOv2)")
    parser.add_argument("--classifier_checkpoint", type=str, required=True,
                        help="DINOv2 failure classifier checkpoint")
    parser.add_argument("--density_checkpoint", type=str, required=True,
                        help="Flow matching density estimator checkpoint")
    parser.add_argument("--mode", type=str, default="ensemble",
                        choices=["baseline", "mc_dropout", "ensemble", "combined", "laplace"])
    parser.add_argument("--pkl", type=str, default=EVAL_FILE)
    parser.add_argument("--episode", type=int, nargs="*", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--calibration", type=str, default=None)
    parser.add_argument("--cal_mode", type=str, default="image_id",
                        choices=["image_id", "image_cd", "traj_id", "traj_cd"])
    parser.add_argument("--output_dir", type=str, default=None)
    # OOD data options
    parser.add_argument("--ood", action="store_true",
                        help="Visualize OOD data instead of eval data")
    parser.add_argument("--ood_dir", type=str, default=str(OOD_DIR),
                        help="Directory with OOD traj_*.pkl files")
    parser.add_argument("--ood_pkl", type=str, nargs="*", default=None,
                        help="Specific OOD pkl file(s) to visualize")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load classifier
    classifier, laplace = load_classifier(
        args.classifier_checkpoint, device, mode=args.mode)

    # Load density model
    density_model, dino_encoder = load_density_model(
        args.density_checkpoint, device)

    # Load calibration threshold
    threshold = None
    if args.calibration:
        with open(args.calibration, "r") as f:
            cal = json.load(f)
        metric_key = "entropy" if args.mode == "baseline" else "variance"
        thresholds = cal.get("thresholds", {})
        if metric_key in thresholds and args.cal_mode in thresholds[metric_key]:
            threshold = thresholds[metric_key][args.cal_mode]["threshold"]
            print(f"Calibration threshold ({args.cal_mode}): {threshold:.6f}")

    uq_key, uq_label = MODE_UQ_METRIC.get(args.mode, ("variances", "Variance"))

    # Load data — either OOD or standard eval
    if args.ood or args.ood_pkl:
        # OOD mode
        if args.ood_pkl:
            episodes = [load_ood_trajectory(p) for p in args.ood_pkl]
            ep_names = [Path(p).stem for p in args.ood_pkl]
        else:
            episodes = load_ood_all(args.ood_dir)
            ep_names = [ep["source"] for ep in episodes]
        print(f"Loaded {len(episodes)} OOD trajectory(ies)")
        indices = list(range(len(episodes)))
        default_output = str(OUTPUT_DIR / "ood" / args.mode)
    else:
        # Standard eval data
        pkl_path = DATA_DIR / args.pkl
        print(f"Loading data: {pkl_path}")
        with open(pkl_path, "rb") as f:
            episodes = pickle.load(f)
        print(f"Loaded {len(episodes)} episodes")
        ep_names = None

        if args.all:
            indices = list(range(len(episodes)))
        elif args.episode is not None:
            indices = args.episode
        else:
            indices = [0]
        default_output = str(OUTPUT_DIR / args.mode)

    # Run inference
    print(f"\nRunning inference on {len(indices)} episode(s)...")
    ep_results = {}
    for idx in indices:
        if idx >= len(episodes):
            print(f"Skipping episode {idx} (out of range)")
            continue
        ep = episodes[idx]
        T = len(ep["failure"])
        tag = ep_names[idx] if ep_names else str(idx)
        print(f"  Episode {tag}: T={T}", end="\r")

        # Classifier
        probs, variances, entropies = run_classifier_episode(
            classifier, ep, args.mode, device, laplace=laplace,
            batch_size=args.batch_size)
        classifier_uq = entropies if uq_key == "entropies" else variances

        # Density
        logpZO = predict_density_episode(
            density_model, dino_encoder, ep, device,
            batch_size=args.batch_size)

        ep_results[idx] = (probs, logpZO, classifier_uq)
    print()

    # Fixed y-axes
    global_uq_max = max(cuq.max() for _, _, cuq in ep_results.values())
    uq_ymax = max(global_uq_max * 1.2, 0.01)
    if threshold is not None:
        uq_ymax = max(uq_ymax, threshold * 1.5)

    global_logp_max = max(logp.max() for _, logp, _ in ep_results.values())
    logpZO_ymax = max(global_logp_max * 1.2, 1.0)

    print(f"UQ y-axis: [0, {uq_ymax:.4f}], logpZO y-axis: [0, {logpZO_ymax:.1f}]")

    # Render
    output_dir = args.output_dir or default_output
    print(f"Rendering {len(ep_results)} video(s)...")
    for idx, (probs, logpZO, classifier_uq) in ep_results.items():
        name_prefix = ep_names[idx] if ep_names else None
        render_episode(episodes[idx], idx, probs, logpZO, classifier_uq,
                       uq_label, args.mode, args.fps, logpZO_ymax, uq_ymax,
                       threshold=threshold, output_dir=output_dir,
                       name_prefix=name_prefix)

    print(f"\nDone! Videos saved to {output_dir}/")


if __name__ == "__main__":
    main()
