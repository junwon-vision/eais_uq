"""
Rendering helpers: single frames, UQ graphs, episode videos, notebook display.
"""

from pathlib import Path

import cv2
import imageio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np

from config import OUTPUT_DIR
from data_utils import decompress_image

BORDER_W = 8

# Mapping from inference mode to the UQ array key and display label.
MODE_UQ_METRIC = {
    "baseline":   ("entropies", "Entropy"),
    "mc_dropout": ("variances", "Variance"),
    "ensemble":   ("variances", "Variance"),
    "combined":   ("variances", "Variance"),
    "laplace":    ("variances", "Variance"),
}

# Fixed y-axis upper limits for each mode (keeps videos comparable across episodes).
MODE_UQ_YMAX = {
    "baseline":   1.0,    # entropy is in [0, ln2 ≈ 0.693]; cap at 1
    "mc_dropout": 0.2,    # variance
    "ensemble":   0.2,    # variance
    "combined":   0.2,    # variance
    "laplace":    0.2,    # logit-variance (different scale, but same cap)
}
DENSITY_YMAX = 3000      # logpZO upper limit


def _get_image(img):
    """Handle both JPEG bytes and raw numpy arrays."""
    if isinstance(img, (bytes, np.bytes_)):
        return decompress_image(img)
    return img


def save_dataset_sample_images(episodes, output_dir=None, episode_idx=0, timestep=0):
    """
    Load dataset from episodes and save two sample images (wrist + 3rd-person/front)
    as .jpg files so they can be loaded in the notebook.
    """
    output_dir = Path(output_dir or OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    ep = episodes[episode_idx]
    wrist_img = _get_image(ep["wrist_cam"][timestep])
    third_img = _get_image(ep["front_cam"][timestep])  # 3rd-person view = front_cam
    wrist_path = output_dir / "dataset_sample_wrist.jpg"
    third_path = output_dir / "dataset_sample_3rd.jpg"
    cv2.imwrite(str(wrist_path), cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(third_path), cv2.cvtColor(third_img, cv2.COLOR_RGB2BGR))
    return wrist_path, third_path


# ---------------------------------------------------------------------------
# Single-row UQ graph
# ---------------------------------------------------------------------------

def render_graph(t, T, uq_scores, mode, graph_width, graph_height,
                 uq_label="UQ", threshold=None, uq_ymax=None):
    """Render single-row UQ graph up to timestep *t* as an RGB numpy array."""
    dpi = 100
    fig_w = graph_width / dpi
    fig_h = graph_height / dpi
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=dpi)

    x_so_far = np.arange(t + 1)
    uq_vals = uq_scores[:t + 1]

    if threshold is not None:
        ax.axhline(y=threshold, color="gray", linestyle=":", linewidth=1.2,
                   alpha=0.8, label=f"threshold={threshold:.4f}")
        for i in range(len(uq_vals) - 1):
            seg_x = [i, i + 1]
            seg_y = [uq_vals[i], uq_vals[i + 1]]
            color = "red" if (uq_vals[i] > threshold or
                              uq_vals[i + 1] > threshold) else "purple"
            ax.plot(seg_x, seg_y, color=color, linewidth=1.8)
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
    ax.set_title(f"Predictive {uq_label} (UQ) \u2014 {mode}", fontsize=9)
    ax.tick_params(labelsize=7)

    fig.tight_layout(pad=0.5)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())[..., :3]  # RGBA → RGB
    plt.close(fig)
    return buf


# ---------------------------------------------------------------------------
# Two-row graph: classifier UQ + density
# ---------------------------------------------------------------------------

def render_graph_density(t, T, logpZO_scores, classifier_uq, mode,
                         graph_width, graph_height, uq_label,
                         logpZO_ymax, uq_ymax, threshold=None):
    """Render 2-row graph (classifier UQ on top, logpZO on bottom)."""
    dpi = 100
    fig_w = graph_width / dpi
    fig_h = graph_height / dpi
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h), dpi=dpi)

    x_so_far = np.arange(t + 1)

    # Row 1: classifier UQ
    ax = axes[0]
    uq_vals = classifier_uq[:t + 1]
    if threshold is not None:
        ax.axhline(y=threshold, color="gray", linestyle=":", linewidth=1.0,
                   alpha=0.8, label=f"\u03c4={threshold:.4f}")
        for i in range(len(uq_vals) - 1):
            seg_x = [i, i + 1]
            seg_y = [uq_vals[i], uq_vals[i + 1]]
            color = ("red" if uq_vals[i] > threshold or
                     uq_vals[i + 1] > threshold else "purple")
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
    ax.set_title(f"Predictive {uq_label} \u2014 {mode}", fontsize=8)
    ax.tick_params(labelsize=6)

    # Row 2: logpZO
    ax = axes[1]
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
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())[..., :3]  # RGBA → RGB
    plt.close(fig)
    return buf


# ---------------------------------------------------------------------------
# Label-only graph (for dataset visualization video)
# ---------------------------------------------------------------------------

def render_label_graph(t, T, failure_labels, graph_width, graph_height):
    """Render a step plot of ground-truth failure labels (0=Success, 1=Failure) up to t."""
    dpi = 100
    fig_w = graph_width / dpi
    fig_h = graph_height / dpi
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=dpi)

    x_so_far = np.arange(t + 1)
    # Map -1 -> 0 for display
    labels = np.array(failure_labels[:t + 1], dtype=np.float64)
    labels[labels < 0] = 0

    ax.fill_between(x_so_far, 0, labels, color="red", alpha=0.4, step="post")
    ax.step(np.arange(t + 2), np.r_[labels, labels[-1]], where="post",
            color="darkred", linewidth=1.5)
    ax.set_xlim(0, T)
    ax.set_ylim(-0.05, 1.15)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Success", "Failure"], fontsize=8)
    ax.set_ylabel("Label", fontsize=8)
    ax.set_xlabel("Timestep", fontsize=8)
    ax.set_title("Ground-truth failure labels", fontsize=9)
    ax.tick_params(labelsize=7)

    fig.tight_layout(pad=0.5)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    return buf


def render_episode_frame_with_labels(ep, t, failure_labels, T, graph_height=200):
    """Compose one frame: header | front+wrist (border by label) | label graph."""
    front = _get_image(ep["front_cam"][t])
    wrist = _get_image(ep["wrist_cam"][t])

    sep = np.zeros((256, 4, 3), dtype=np.uint8)
    cam_row = np.concatenate([front, sep, wrist], axis=1)

    lbl = int(failure_labels[t]) if failure_labels[t] >= 0 else 0
    border_color = (np.array([255, 0, 0], dtype=np.uint8) if lbl == 1
                    else np.array([0, 255, 0], dtype=np.uint8))
    cam_row[:BORDER_W, :] = border_color
    cam_row[-BORDER_W:, :] = border_color
    cam_row[:, :BORDER_W] = border_color
    cam_row[:, -BORDER_W:] = border_color

    cam_w = cam_row.shape[1]

    header_h = 32
    header = np.zeros((header_h, cam_w, 3), dtype=np.uint8) + 30
    header_bgr = header[:, :, ::-1].copy()
    label_str = "Failure" if lbl == 1 else "Success"
    label_color = (60, 60, 255) if lbl == 1 else (60, 200, 60)
    cv2.putText(header_bgr, f"t={t}/{T}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(header_bgr, label_str, (120, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)
    header = header_bgr[:, :, ::-1].copy()

    graph = render_label_graph(t, T, failure_labels, cam_w, graph_height)
    if graph.shape[1] != cam_w:
        graph = cv2.resize(graph, (cam_w, graph_height))

    frame = np.concatenate([header, cam_row, graph], axis=0)
    return frame


def save_episode_with_labels_mp4(episodes, episode_idx=0, output_path=None, fps=10):
    """
    Save one episode as mp4 with front+wrist frames and a graph of ground-truth
    failure labels. No model; uses only dataset labels.
    """
    ep = episodes[episode_idx]
    failure = ep["failure"]
    T = len(failure)

    frames = []
    for t in range(T):
        frame = render_episode_frame_with_labels(ep, t, failure, T)
        frames.append(frame)

    if output_path is None:
        save_dir = OUTPUT_DIR / "videos"
        save_dir.mkdir(parents=True, exist_ok=True)
        output_path = save_dir / f"episode_labels_ep{episode_idx:04d}_T{T}.mp4"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    imageio.mimwrite(str(output_path), frames, fps=fps, codec="libx264",
                     macro_block_size=1,
                     output_params=["-pix_fmt", "yuv420p"])
    return str(output_path)


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def render_frame(ep, t, probs, uq_scores, uq_label, mode, T,
                 threshold=None, uq_ymax=None):
    """
    Compose a full video frame:
      header | front+wrist cameras (with colored border) | UQ graph
    """
    front = _get_image(ep["front_cam"][t])
    wrist = _get_image(ep["wrist_cam"][t])

    # Invert R/B channels when UQ above threshold
    if threshold is not None and uq_scores[t] > threshold:
        front = front[:, :, ::-1].copy()
        wrist = wrist[:, :, ::-1].copy()

    sep = np.zeros((256, 4, 3), dtype=np.uint8)
    cam_row = np.concatenate([front, sep, wrist], axis=1)

    # Colored border: green (OK) / red (FAILURE)
    prob_now = probs[t]
    border_color = (np.array([255, 0, 0], dtype=np.uint8) if prob_now > 0.5
                    else np.array([0, 255, 0], dtype=np.uint8))
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
    header = header_bgr[:, :, ::-1].copy()

    # UQ graph
    graph_h = 200
    graph = render_graph(t, T, uq_scores, mode,
                         graph_width=cam_w, graph_height=graph_h,
                         uq_label=uq_label, threshold=threshold,
                         uq_ymax=uq_ymax)
    if graph.shape[1] != cam_w:
        graph = cv2.resize(graph, (cam_w, graph_h))

    frame = np.concatenate([header, cam_row, graph], axis=0)
    return frame


def render_frame_no_graph(ep, t, probs, uq_scores, uq_label, mode, T,
                          threshold=None):
    """
    Render a frame with only header + camera views (no UQ graph).
    """
    front = _get_image(ep["front_cam"][t])
    wrist = _get_image(ep["wrist_cam"][t])

    if threshold is not None and uq_scores[t] > threshold:
        front = front[:, :, ::-1].copy()
        wrist = wrist[:, :, ::-1].copy()

    sep = np.zeros((256, 4, 3), dtype=np.uint8)
    cam_row = np.concatenate([front, sep, wrist], axis=1)

    prob_now = probs[t]
    border_color = (np.array([255, 0, 0], dtype=np.uint8) if prob_now > 0.5
                    else np.array([0, 255, 0], dtype=np.uint8))
    cam_row[:BORDER_W, :] = border_color
    cam_row[-BORDER_W:, :] = border_color
    cam_row[:, :BORDER_W] = border_color
    cam_row[:, -BORDER_W:] = border_color

    cam_w = cam_row.shape[1]

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
    header = header_bgr[:, :, ::-1].copy()

    frame = np.concatenate([header, cam_row], axis=0)
    return frame


# ---------------------------------------------------------------------------
# Episode video
# ---------------------------------------------------------------------------

def render_episode_video(ep, ep_idx, probs, uq_scores, uq_label, mode,
                         fps=10, threshold=None, uq_ymax=None,
                         output_path=None):
    """
    Render all frames of an episode and write to mp4.

    Returns the path to the written file.
    """
    T = len(probs)
    if uq_ymax is None:
        uq_ymax = max(float(uq_scores.max()) * 1.2, 0.01)
        if threshold is not None:
            uq_ymax = max(uq_ymax, threshold * 1.5)

    frames = []
    for t in range(T):
        frame = render_frame(ep, t, probs, uq_scores, uq_label, mode, T,
                             threshold=threshold, uq_ymax=uq_ymax)
        frames.append(frame)

    if output_path is None:
        save_dir = OUTPUT_DIR / "videos"
        save_dir.mkdir(parents=True, exist_ok=True)
        output_path = save_dir / f"ep{ep_idx:04d}_{mode}_uq_T{T}.mp4"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    imageio.mimwrite(str(output_path), frames, fps=fps, codec="libx264",
                     macro_block_size=1,
                     output_params=["-pix_fmt", "yuv420p"])
    return str(output_path)


def render_episode_video_no_graph(ep, ep_idx, probs, uq_scores, uq_label, mode,
                                  fps=10, threshold=None,
                                  output_path=None):
    """
    Render all frames of an episode with only header + camera views (no UQ graph).

    Returns the path to the written file.
    """
    T = len(probs)

    frames = []
    for t in range(T):
        frame = render_frame_no_graph(ep, t, probs, uq_scores, uq_label, mode, T,
                                      threshold=threshold)
        frames.append(frame)

    if output_path is None:
        save_dir = OUTPUT_DIR / "videos"
        save_dir.mkdir(parents=True, exist_ok=True)
        output_path = save_dir / f"ep{ep_idx:04d}_{mode}_imgonly_T{T}.mp4"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    imageio.mimwrite(str(output_path), frames, fps=fps, codec="libx264",
                     macro_block_size=1,
                     output_params=["-pix_fmt", "yuv420p"])
    return str(output_path)


# ---------------------------------------------------------------------------
# Density-only graph (single row: logpZO)
# ---------------------------------------------------------------------------

def render_logpZO_graph(t, T, logpZO_scores, graph_width, graph_height,
                        logpZO_ymax=None, threshold=None):
    """Render single-row graph of logpZO (density) up to timestep t."""
    if logpZO_ymax is None:
        logpZO_ymax = max(float(np.array(logpZO_scores[:t + 1]).max()) * 1.2, 1.0)
    dpi = 100
    fig_w = graph_width / dpi
    fig_h = graph_height / dpi
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=dpi)
    x_so_far = np.arange(t + 1)
    logp_vals = logpZO_scores[:t + 1]
    ax.plot(x_so_far, logp_vals, color="darkgreen", linewidth=1.5)
    ax.fill_between(x_so_far, 0, logp_vals, color="green", alpha=0.1)
    if threshold is not None:
        ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2,
                   label=f"τ={threshold:.1f}")
        ax.legend(fontsize=7, loc="upper left")
    ax.set_xlim(0, T)
    ax.set_ylim(0, logpZO_ymax)
    ax.set_ylabel("logpZO", fontsize=8)
    ax.set_xlabel("Timestep", fontsize=8)
    ax.set_title("Density (higher = more OOD)", fontsize=9)
    ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.5)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    return buf


# ---------------------------------------------------------------------------
# Density-only episode video (no classifier UQ in graph)
# ---------------------------------------------------------------------------

def render_episode_video_density(ep, ep_idx, probs, logpZO_scores,
                                 classifier_uq, uq_label, mode,
                                 fps=10, threshold=None,
                                 uq_ymax=None, logpZO_ymax=None,
                                 output_path=None):
    """Render episode video with density-only graph (logpZO). Classifier still used for header/border."""
    T = len(probs)
    if logpZO_ymax is None:
        logpZO_ymax = max(float(np.array(logpZO_scores).max()) * 1.2, 1.0)

    frames = []
    for t in range(T):
        front = _get_image(ep["front_cam"][t])
        wrist = _get_image(ep["wrist_cam"][t])

        if threshold is not None and classifier_uq[t] > threshold:
            front = front[:, :, ::-1].copy()
            wrist = wrist[:, :, ::-1].copy()

        sep = np.zeros((256, 4, 3), dtype=np.uint8)
        cam_row = np.concatenate([front, sep, wrist], axis=1)

        prob_now = probs[t]
        border_color = (np.array([255, 0, 0], dtype=np.uint8) if prob_now > 0.5
                        else np.array([0, 255, 0], dtype=np.uint8))
        cam_row[:BORDER_W, :] = border_color
        cam_row[-BORDER_W:, :] = border_color
        cam_row[:, :BORDER_W] = border_color
        cam_row[:, -BORDER_W:] = border_color

        cam_w = cam_row.shape[1]

        # Header
        header_h = 32
        header = np.zeros((header_h, cam_w, 3), dtype=np.uint8) + 30
        header_bgr = header[:, :, ::-1].copy()
        logp_now = logpZO_scores[t]
        pred = "FAILURE" if prob_now > 0.5 else "OK"
        pred_color = (60, 60, 255) if prob_now > 0.5 else (60, 200, 60)
        cv2.putText(header_bgr, f"t={t}/{T}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        cv2.putText(header_bgr, pred, (120, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, pred_color, 2)
        cv2.putText(header_bgr, f"P={prob_now:.3f}  logpZO={logp_now:.1f}",
                    (230, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1)
        header = header_bgr[:, :, ::-1].copy()

        graph_h = 200
        graph = render_logpZO_graph(
            t, T, logpZO_scores,
            graph_width=cam_w, graph_height=graph_h,
            logpZO_ymax=logpZO_ymax, threshold=threshold)
        if graph.shape[1] != cam_w:
            graph = cv2.resize(graph, (cam_w, graph_h))

        frame = np.concatenate([header, cam_row, graph], axis=0)
        frames.append(frame)

    if output_path is None:
        save_dir = OUTPUT_DIR / "videos"
        save_dir.mkdir(parents=True, exist_ok=True)
        output_path = save_dir / f"ep{ep_idx:04d}_{mode}_density_uq_T{T}.mp4"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    imageio.mimwrite(str(output_path), frames, fps=fps, codec="libx264",
                     macro_block_size=1,
                     output_params=["-pix_fmt", "yuv420p"])
    return str(output_path)


# ---------------------------------------------------------------------------
# Notebook display helpers
# ---------------------------------------------------------------------------

def display_video(path):
    """Display an mp4 video inline in a Jupyter notebook."""
    from IPython.display import Video, display as ipy_display
    ipy_display(Video(str(path), embed=True))


def visualize_single_image(model, ep, t, mode, device,
                           laplace=None, uq_fn=None):
    """
    Render a single-image matplotlib figure showing both cameras,
    the prediction, and UQ score.

    *uq_fn* should be a callable (model, front, wrist, mode, device) → dict
    with keys "prob", "variance", "entropy".  If None, uses predict_single.
    """
    from uq_inference import predict_single
    front_bytes = ep["front_cam"][t]
    wrist_bytes = ep["wrist_cam"][t]

    if uq_fn is None:
        result = predict_single(model, front_bytes, wrist_bytes,
                                mode=mode, device=device)
    else:
        result = uq_fn(model, front_bytes, wrist_bytes, mode, device)

    front_img = _get_image(front_bytes)
    wrist_img = _get_image(wrist_bytes)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(front_img)
    axes[0].set_title("Front cam")
    axes[0].axis("off")
    axes[1].imshow(wrist_img)
    axes[1].set_title("Wrist cam")
    axes[1].axis("off")

    pred = "FAILURE" if result["prob"] > 0.5 else "OK"
    fig.suptitle(f"Prediction: {pred}  |  P(fail)={result['prob']:.3f}  |  "
                 f"Var={result['variance']:.5f}  Ent={result['entropy']:.4f}",
                 fontsize=10)
    fig.tight_layout()
    return fig


def visualize_episode_grid(ep, indices=None, n_frames=8):
    """
    Show a grid of sampled frames (front + wrist) with failure-label timeline.
    """
    from data_utils import sample_frames

    if indices is None:
        indices = sample_frames(ep, n_frames=n_frames)

    failure = ep["failure"]
    T = len(failure)

    fig, axes = plt.subplots(2, len(indices), figsize=(len(indices) * 2.5, 5))
    if len(indices) == 1:
        axes = axes[:, np.newaxis]

    for col, t in enumerate(indices):
        axes[0, col].imshow(_get_image(ep["front_cam"][t]))
        lbl = failure[t]
        color = "red" if lbl == 1 else ("green" if lbl == 0 else "gray")
        axes[0, col].set_title(f"t={t}", fontsize=8, color=color)
        axes[0, col].axis("off")

        axes[1, col].imshow(_get_image(ep["wrist_cam"][t]))
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Front", fontsize=10)
    axes[1, 0].set_ylabel("Wrist", fontsize=10)
    fig.suptitle(f"Episode (T={T})", fontsize=11)
    fig.tight_layout()
    return fig
