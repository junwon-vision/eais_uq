#!/usr/bin/env python3
"""
Density estimation UQ using flow matching on DINOv2 features.

Trains a ConditionalUnet1D to learn the vector field from the DINOv2 feature
distribution to a standard Gaussian via flow matching. At inference, the
negative log-density (logpZO) serves as an uncertainty score: higher values
indicate the input is far from the training distribution (OOD).

Flow matching training:
    x0 = data feature,  x1 ~ N(0,I)
    v_true = x1 - x0
    x_t = x0 + t * v_true,  t ~ U(0,1)
    loss = ||v_hat(x_t, t) - v_true||^2

Inference (log-density proxy):
    v_hat = net(x, t=0)
    z = x + v_hat            (push to Gaussian)
    logpZO = ||z||^2          (higher = lower density = more uncertain)

Usage:
    python eai_safety_hw/train_density_dinov2.py train \
        --checkpoint_dir checkpoints/density_dinov2 --epochs 50

    python eai_safety_hw/train_density_dinov2.py eval \
        --checkpoint checkpoints/density_dinov2/checkpoint.pt

    python eai_safety_hw/train_density_dinov2.py extract_features \
        --checkpoint_dir checkpoints/density_dinov2
"""

import argparse
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eai_safety_hw.diffusion_policy.model.diffusion.conditional_unet1d import (
    ConditionalUnet1D,
)

DATA_DIR = Path("/home/junwon/uq_sqfety/datasets/dreamer_fixed")
DEFAULT_CKPT_DIR = Path("checkpoints/density_dinov2")

TRAIN_FILES = ["success_only.pkl", "failure_labeled.pkl"]
EVAL_FILE = "failure_eval.pkl"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def decompress_image(compressed_img):
    encoded = np.frombuffer(compressed_img, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    assert bgr.shape[:2] == (256, 256), f"Unexpected shape: {bgr.shape}"
    return bgr[:, :, ::-1].copy()


def load_episodes(filenames):
    episodes = []
    for fname in filenames:
        path = DATA_DIR / fname
        if not path.exists():
            print(f"Warning: {path} not found, skipping")
            continue
        with open(path, "rb") as f:
            eps = pickle.load(f)
        print(f"  Loaded {len(eps)} episodes from {fname}")
        episodes.extend(eps)
    return episodes


_img_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# DINOv2 feature extractor (frozen)
# ---------------------------------------------------------------------------

class DINOv2Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14", pretrained=True,
        )
        self.embed_dim = self.backbone.embed_dim  # 384
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            return self.backbone(x)


@torch.no_grad()
def extract_features_from_episodes(episodes, device, batch_size=64):
    """Extract DINOv2 features for all timesteps across episodes.

    Returns:
        features: (N, 768) tensor of concatenated [front, wrist] features
        labels:   (N,) array of failure labels
        ep_boundaries: list of (start, end) indices per episode
    """
    encoder = DINOv2Encoder().to(device)
    encoder.eval()

    all_features = []
    all_labels = []
    ep_boundaries = []
    idx = 0

    for ep_i, ep in enumerate(episodes):
        failure = ep["failure"]
        T = len(failure)
        front_list = ep["front_cam"]
        wrist_list = ep["wrist_cam"]

        ep_feats = []
        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            fronts, wrists = [], []
            for t in range(start, end):
                fronts.append(_img_transform(decompress_image(front_list[t])))
                wrists.append(_img_transform(decompress_image(wrist_list[t])))
            fronts = torch.stack(fronts).to(device)
            wrists = torch.stack(wrists).to(device)

            f_feat = encoder(fronts)  # (B, 384)
            w_feat = encoder(wrists)  # (B, 384)
            combined = torch.cat([f_feat, w_feat], dim=1)  # (B, 768)
            ep_feats.append(combined.cpu())

        ep_feats = torch.cat(ep_feats, dim=0)  # (T, 768)
        labels = np.array(failure, dtype=np.float32)
        labels[labels < 0] = 0.0  # -1 -> 0

        all_features.append(ep_feats)
        all_labels.append(labels)
        ep_boundaries.append((idx, idx + T))
        idx += T

        if (ep_i + 1) % 10 == 0 or ep_i == len(episodes) - 1:
            print(f"  Extracted features: {ep_i+1}/{len(episodes)} episodes", end="\r")

    print()
    features = torch.cat(all_features, dim=0)
    labels = np.concatenate(all_labels)
    return features, labels, ep_boundaries


# ---------------------------------------------------------------------------
# Flow matching density estimator (logpZO)
# ---------------------------------------------------------------------------

def build_unet(input_dim):
    """Build ConditionalUnet1D for flow matching."""
    return ConditionalUnet1D(
        input_dim=input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=128,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=False,
    )


class FlowMatchingDensity(nn.Module):
    """
    Flow matching density estimator on feature vectors.

    Training: learns vector field v(x,t) that maps data -> Gaussian.
    Inference: pushes x through v at t=0, computes ||z||^2 as neg-log-density.
    """

    def __init__(self, input_dim, device="cuda"):
        super().__init__()
        self.input_dim = input_dim
        self.net = build_unet(input_dim)
        self.time_scale = 100  # UNet expects discrete time steps

    def train_step(self, features):
        """
        One training step of flow matching.

        Args:
            features: (B, D) tensor of DINOv2 features

        Returns:
            loss: scalar tensor
        """
        self.net.train()
        device = features.device
        B = features.shape[0]

        # Reshape for UNet: (B, 1, D) where T=1 (single "timestep" in sequence dim)
        x0 = features.reshape(B, 1, self.input_dim)
        x1 = torch.randn_like(x0)

        v_true = x1 - x0

        # Random interpolation time t ~ U(0,1)
        t = torch.rand(B, device=device)
        t_expand = t.view(B, 1, 1)
        x_t = x0 + t_expand * v_true

        # UNet predicts velocity at (x_t, t)
        v_hat = self.net(x_t, (t * self.time_scale).long())

        loss = (v_hat - v_true).pow(2).mean()
        return loss

    @torch.no_grad()
    def compute_logpZO(self, features):
        """
        Compute log-density proxy for input features.

        Args:
            features: (B, D) tensor

        Returns:
            logpZO: (B,) tensor — higher = lower density = more uncertain
        """
        self.net.eval()
        B = features.shape[0]

        x = features.reshape(B, 1, self.input_dim)
        timesteps = torch.zeros(B, device=features.device, dtype=torch.long)
        v_hat = self.net(x, timesteps)

        z = x + v_hat  # push to Gaussian
        logpZO = z.reshape(B, -1).pow(2).sum(dim=-1)
        return logpZO


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_extract_features(args):
    """Extract and cache DINOv2 features for training data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("Loading training episodes...")
    episodes = load_episodes(TRAIN_FILES)
    print(f"Total episodes: {len(episodes)}")

    print("Extracting DINOv2 features...")
    features, labels, ep_boundaries = extract_features_from_episodes(
        episodes, device, batch_size=args.batch_size)

    cache_path = ckpt_dir / "train_features.pt"
    torch.save({
        "features": features,
        "labels": labels,
        "ep_boundaries": ep_boundaries,
    }, cache_path)
    print(f"Saved features: {cache_path} ({features.shape})")


def run_train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Try to load cached features
    cache_path = ckpt_dir / "train_features.pt"
    if cache_path.exists():
        print(f"Loading cached features from {cache_path}")
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        features = cached["features"]
        labels = cached["labels"]
    else:
        print("No cached features found. Extracting DINOv2 features...")
        episodes = load_episodes(TRAIN_FILES)
        features, labels, _ = extract_features_from_episodes(
            episodes, device, batch_size=args.batch_size)
        torch.save({
            "features": features,
            "labels": labels,
        }, cache_path)
        print(f"Cached features to {cache_path}")

    input_dim = features.shape[1]  # 768
    print(f"Training features: {features.shape}, input_dim={input_dim}")
    print(f"Label distribution: success={int((labels == 0).sum())}, "
          f"failure={int((labels == 1).sum())}")

    # DataLoader
    dataset = TensorDataset(features)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        drop_last=True, num_workers=0)

    # Model
    model = FlowMatchingDensity(input_dim, device=device).to(device)
    n_params = sum(p.numel() for p in model.net.parameters())
    print(f"UNet parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.net.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float("inf")
    best_state = None

    for epoch in range(args.epochs):
        model.net.train()
        epoch_loss = 0.0
        n_batches = 0

        for (batch_feats,) in tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            batch_feats = batch_feats.to(device)
            optimizer.zero_grad()
            loss = model.train_step(batch_feats)
            loss.backward()
            nn.utils.clip_grad_norm_(model.net.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches

        # Compute density stats on a subset
        model.net.eval()
        with torch.no_grad():
            subset = features[:min(2000, len(features))].to(device)
            logpZO_vals = model.compute_logpZO(subset).cpu().numpy()
            sub_labels = labels[:min(2000, len(labels))]

        success_logp = logpZO_vals[sub_labels == 0]
        failure_logp = logpZO_vals[sub_labels == 1]

        print(f"  Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.6f} | "
              f"logpZO: mean={logpZO_vals.mean():.2f}, std={logpZO_vals.std():.2f}")
        if len(failure_logp) > 0:
            print(f"    Success: {success_logp.mean():.2f} +/- {success_logp.std():.2f} | "
                  f"Failure: {failure_logp.mean():.2f} +/- {failure_logp.std():.2f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save checkpoint
    ckpt_path = ckpt_dir / "checkpoint.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": input_dim,
        "args": vars(args),
    }, ckpt_path)
    print(f"\nSaved checkpoint: {ckpt_path}")


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def run_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    input_dim = ckpt["input_dim"]

    model = FlowMatchingDensity(input_dim, device=device).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Loaded density model (input_dim={input_dim})")

    # Extract eval features
    print("Loading eval data...")
    episodes = load_episodes([EVAL_FILE])
    print(f"Eval episodes: {len(episodes)}")

    print("Extracting DINOv2 features...")
    features, labels, ep_boundaries = extract_features_from_episodes(
        episodes, device, batch_size=args.batch_size)

    # Compute logpZO
    print("Computing log-density scores...")
    all_logpZO = []
    for start in range(0, len(features), args.batch_size):
        end = min(start + args.batch_size, len(features))
        batch = features[start:end].to(device)
        logpZO = model.compute_logpZO(batch).cpu().numpy()
        all_logpZO.extend(logpZO)

    all_logpZO = np.array(all_logpZO)

    # Metrics
    success_mask = labels == 0
    failure_mask = labels == 1

    print(f"\n{'='*50}")
    print(f"Density Estimation (logpZO) Results")
    print(f"{'='*50}")
    print(f"  Total samples: {len(all_logpZO)}")
    print(f"  logpZO overall: {all_logpZO.mean():.4f} +/- {all_logpZO.std():.4f}")
    if success_mask.sum() > 0:
        print(f"  logpZO success: {all_logpZO[success_mask].mean():.4f} +/- "
              f"{all_logpZO[success_mask].std():.4f}")
    if failure_mask.sum() > 0:
        print(f"  logpZO failure: {all_logpZO[failure_mask].mean():.4f} +/- "
              f"{all_logpZO[failure_mask].std():.4f}")

    # Per-episode
    print(f"\n  Per-episode (max logpZO):")
    for i, (s, e) in enumerate(ep_boundaries):
        ep_logp = all_logpZO[s:e]
        ep_labels = labels[s:e]
        ep_is_fail = "FAIL" if ep_labels.max() > 0.5 else "OK"
        print(f"    Ep {i:3d}: max={ep_logp.max():.2f}, mean={ep_logp.mean():.2f}, "
              f"label={ep_is_fail}")
        if i >= 19:
            print(f"    ... ({len(ep_boundaries) - 20} more)")
            break

    # AUROC using logpZO as anomaly score
    if failure_mask.sum() > 0 and success_mask.sum() > 0:
        from sklearn.metrics import roc_auc_score
        try:
            auroc = roc_auc_score(labels, all_logpZO)
            print(f"\n  AUROC (logpZO as anomaly detector): {auroc:.4f}")
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# Inference API (for visualization scripts)
# ---------------------------------------------------------------------------

def load_density_model(checkpoint_path, device):
    """Load trained density model. Returns (model, dino_encoder)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    input_dim = ckpt["input_dim"]
    model = FlowMatchingDensity(input_dim, device=device).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    encoder = DINOv2Encoder().to(device)
    encoder.eval()

    print(f"Loaded density model (input_dim={input_dim})")
    return model, encoder


@torch.no_grad()
def predict_density_episode(model, encoder, ep, device, batch_size=32):
    """
    Run density estimation on all timesteps of an episode.

    Args:
        model: FlowMatchingDensity model
        encoder: DINOv2Encoder
        ep: episode dict with front_cam, wrist_cam, failure keys
            (images can be either JPEG bytes or raw numpy arrays)

    Returns:
        logpZO: (T,) array — higher = more OOD / uncertain
    """
    T = len(ep["failure"])
    all_logpZO = []

    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        fronts, wrists = [], []
        for t in range(start, end):
            f_img = ep["front_cam"][t]
            w_img = ep["wrist_cam"][t]
            # Handle both JPEG bytes and raw numpy arrays
            if isinstance(f_img, (bytes, np.bytes_)):
                f_img = decompress_image(f_img)
            if isinstance(w_img, (bytes, np.bytes_)):
                w_img = decompress_image(w_img)
            fronts.append(_img_transform(f_img))
            wrists.append(_img_transform(w_img))

        fronts = torch.stack(fronts).to(device)
        wrists = torch.stack(wrists).to(device)

        f_feat = encoder(fronts)
        w_feat = encoder(wrists)
        combined = torch.cat([f_feat, w_feat], dim=1)  # (B, 768)

        logpZO = model.compute_logpZO(combined).cpu().numpy()
        all_logpZO.extend(logpZO)

    return np.array(all_logpZO)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Density estimation UQ using flow matching on DINOv2 features")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # extract_features
    ext_parser = subparsers.add_parser("extract_features",
        help="Pre-extract and cache DINOv2 features")
    ext_parser.add_argument("--checkpoint_dir", type=str,
                            default=str(DEFAULT_CKPT_DIR))
    ext_parser.add_argument("--batch_size", type=int, default=64)

    # train
    train_parser = subparsers.add_parser("train",
        help="Train flow matching density estimator")
    train_parser.add_argument("--checkpoint_dir", type=str,
                              default=str(DEFAULT_CKPT_DIR))
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch_size", type=int, default=256)
    train_parser.add_argument("--lr", type=float, default=1e-4)

    # eval
    eval_parser = subparsers.add_parser("eval",
        help="Evaluate density estimator")
    eval_parser.add_argument("--checkpoint", type=str, required=True)
    eval_parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()

    if args.command == "extract_features":
        run_extract_features(args)
    elif args.command == "train":
        run_train(args)
    elif args.command == "eval":
        run_eval(args)


if __name__ == "__main__":
    main()
