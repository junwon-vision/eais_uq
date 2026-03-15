#!/usr/bin/env python3
"""
Failure classifier using frozen DINOv2 ViT-S/14 backbone with MC Dropout / Ensemble UQ.

Backbone: dinov2_vits14 (ViT-Small, patch size 14, 384-dim CLS token, ~22M params frozen)
Only the MLP head is trained. Ensemble diversity and MC Dropout are applied to the head only.

Usage:
    python train_failure_classifier_dinov2.py inspect
    python train_failure_classifier_dinov2.py train --ensemble_size 1 --dropout_rate 0.0   # baseline
    python train_failure_classifier_dinov2.py train --ensemble_size 1 --dropout_rate 0.2   # mc dropout
    python train_failure_classifier_dinov2.py train --ensemble_size 10 --dropout_rate 0.0  # ensemble
    python train_failure_classifier_dinov2.py eval --checkpoint <path> --mode ensemble
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
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

DATA_DIR = Path("/home/junwon/uq_sqfety/datasets/dreamer_fixed")
DEFAULT_CKPT_DIR = Path("checkpoints/failure_classifier_dinov2")

# TRAIN_FILES = ["success_only.pkl", "failure_labeled.pkl"]
TRAIN_FILES = ["failure_labeled.pkl"]
EVAL_FILE = "failure_eval.pkl"

# DINOv2 uses ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def decompress_image(compressed_img):
    """Decompress JPEG bytes to RGB uint8 numpy array (256x256x3)."""
    encoded = np.frombuffer(compressed_img, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    assert bgr.shape[:2] == (256, 256), f"Unexpected shape: {bgr.shape}"
    return bgr[:, :, ::-1].copy()  # BGR -> RGB


def load_episodes(filenames):
    """Load episodes from pkl files. Returns list of episode dicts."""
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


# DINOv2 preprocessing: resize to 224 (patch_size=14 → 16x16 patches), ImageNet normalize
_img_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class FailureDataset(Dataset):
    """Per-timestep dataset: each sample is (front_img, wrist_img, label)."""

    def __init__(self, episodes):
        self.samples = []  # (front_bytes, wrist_bytes, label)
        for ep in episodes:
            failure = ep["failure"]
            front_list = ep["front_cam"]
            wrist_list = ep["wrist_cam"]
            for t in range(len(failure)):
                label = failure[t]
                if label < 0:
                    label = 0.0
                self.samples.append((front_list[t], wrist_list[t], float(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        front_bytes, wrist_bytes, label = self.samples[idx]
        front_img = decompress_image(front_bytes)
        wrist_img = decompress_image(wrist_bytes)
        front = _img_transform(front_img)
        wrist = _img_transform(wrist_img)
        return front, wrist, torch.tensor(label, dtype=torch.float32)

    def get_labels(self):
        return [s[2] for s in self.samples]


def make_weighted_sampler(dataset):
    """WeightedRandomSampler with inverse-frequency weights for class balance."""
    labels = dataset.get_labels()
    labels_arr = np.array(labels)
    class_counts = np.bincount(labels_arr.astype(int), minlength=2)
    weights_per_class = 1.0 / (class_counts + 1e-6)
    sample_weights = weights_per_class[labels_arr.astype(int)]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(dataset),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DINOv2Encoder(nn.Module):
    """
    Frozen DINOv2 ViT-S/14 feature extractor.
    Input:  (B, 3, 224, 224) ImageNet-normalized
    Output: (B, 384) CLS token embedding
    """

    def __init__(self, freeze=True):
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14", pretrained=True,
        )
        self.embed_dim = self.backbone.embed_dim  # 384
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        if not any(p.requires_grad for p in self.backbone.parameters()):
            with torch.no_grad():
                return self.backbone(x)
        return self.backbone(x)


def init_head_weights(head, seed=None):
    """Initialize MLP head with Kaiming init. Different seed per ensemble member."""
    if seed is not None:
        torch.manual_seed(seed)
    for m in head.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)


class FailureClassifierDINOv2(nn.Module):
    """
    Binary failure classifier with frozen DINOv2 backbone + trainable MLP head.

    Architecture:
        front_cam (256x256) -> resize 224 -> DINOv2 ViT-S/14 (frozen) -> 384-dim
        wrist_cam (256x256) -> resize 224 -> DINOv2 ViT-S/14 (frozen) -> 384-dim
                                                    |
                                              concat -> 768-dim
                                                    |
                                    Linear(768, 512)  + ReLU + Dropout
                                    Linear(512, 256)  + ReLU + Dropout
                                    Linear(256, 128)  + ReLU + Dropout
                                    Linear(128, 128)  + ReLU + Dropout
                                    Linear(128, 1)    -> logit
    """

    def __init__(self, dropout_rate=0.2, freeze_backbone=True):
        super().__init__()
        self.front_encoder = DINOv2Encoder(freeze=freeze_backbone)
        self.wrist_encoder = DINOv2Encoder(freeze=freeze_backbone)
        embed_dim = self.front_encoder.embed_dim  # 384
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),  # 768 -> 512
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),            # 512 -> 256
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),            # 256 -> 128
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 128),            # 128 -> 128
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),              # 128 -> 1
        )

    def get_features(self, front, wrist):
        """Extract deterministic features before the last linear layer (128-dim)."""
        assert not self.training, "get_features() requires eval mode"
        f_feat = self.front_encoder(front)
        w_feat = self.wrist_encoder(wrist)
        combined = torch.cat([f_feat, w_feat], dim=1)  # (B, 768)
        return self.head[:12](combined)  # (B, 128)

    @property
    def last_layer(self):
        """The final Linear(128, 1) layer."""
        return self.head[12]

    def forward(self, front, wrist):
        f_feat = self.front_encoder(front)  # (B, 384)
        w_feat = self.wrist_encoder(wrist)  # (B, 384)
        combined = torch.cat([f_feat, w_feat], dim=1)  # (B, 768)
        return self.head(combined).squeeze(-1)  # (B,) logits


class FailureClassifierDINOv2Ensemble(nn.Module):
    """Ensemble of K FailureClassifierDINOv2 instances with shared backbone."""

    def __init__(self, ensemble_size=5, dropout_rate=0.2, freeze_backbone=True):
        super().__init__()
        self.members = nn.ModuleList()
        for i in range(ensemble_size):
            member = FailureClassifierDINOv2(dropout_rate, freeze_backbone)
            init_head_weights(member.head, seed=i * 9999 + 7)
            self.members.append(member)

    def forward(self, front, wrist):
        """Returns stacked logits: (K, B)."""
        return torch.stack([m(front, wrist) for m in self.members], dim=0)


# ---------------------------------------------------------------------------
# Last-Layer Laplace Approximation
# ---------------------------------------------------------------------------

class LaplaceWrapper:
    """Last-Layer Laplace Approximation for FailureClassifierDINOv2."""

    def __init__(self, model, prior_precision=1.0):
        self.model = model
        self.prior_precision = prior_precision
        self.posterior_cov = None
        self.fitted = False

    def fit(self, dataloader, device="cuda"):
        self.model.eval()
        self.model.to(device)

        D = self.model.last_layer.in_features  # 256
        H = torch.zeros(D + 1, D + 1, device=device)

        n_samples = 0
        with torch.no_grad():
            for front, wrist, _ in dataloader:
                front, wrist = front.to(device), wrist.to(device)
                B = front.size(0)

                phi = self.model.get_features(front, wrist)  # (B, D)
                phi_aug = torch.cat([phi, torch.ones(B, 1, device=device)], dim=1)

                logits = self.model.last_layer(phi).squeeze(-1)
                probs = torch.sigmoid(logits)

                weights = probs * (1 - probs)
                weighted_phi = phi_aug * weights.unsqueeze(1).sqrt()
                H += weighted_phi.T @ weighted_phi
                n_samples += B

        posterior_precision = H + self.prior_precision * torch.eye(D + 1, device=device)
        self.posterior_cov = torch.linalg.inv(posterior_precision)
        self.fitted = True
        print(f"  Laplace fitted on {n_samples} samples, "
              f"prior_precision={self.prior_precision}, "
              f"cov trace={self.posterior_cov.trace().item():.6f}")

    def predict(self, front, wrist):
        assert self.fitted, "Call .fit() first"
        device = front.device

        self.model.eval()
        with torch.no_grad():
            phi = self.model.get_features(front, wrist)
            B, D = phi.shape
            phi_aug = torch.cat([phi, torch.ones(B, 1, device=device)], dim=1)

            logits = self.model.last_layer(phi).squeeze(-1)
            logit_vars = (phi_aug @ self.posterior_cov * phi_aug).sum(dim=1)
            kappa = 1.0 / torch.sqrt(1.0 + (torch.pi / 8.0) * logit_vars)
            probs = torch.sigmoid(logits * kappa)

        probs_np = probs.cpu().numpy()
        logit_vars_np = logit_vars.cpu().numpy()
        p = np.clip(probs_np, 1e-8, 1 - 1e-8)
        entropies = -p * np.log(p) - (1 - p) * np.log(1 - p)
        return probs_np, logit_vars_np, entropies

    def save(self, path):
        torch.save({
            "posterior_cov": self.posterior_cov.cpu(),
            "prior_precision": self.prior_precision,
        }, path)
        print(f"  Laplace state saved to {path}")

    def load(self, path, device="cuda"):
        state = torch.load(path, map_location=device, weights_only=False)
        self.posterior_cov = state["posterior_cov"].to(device)
        self.prior_precision = state["prior_precision"]
        self.fitted = True
        print(f"  Laplace state loaded from {path}")


def batch_predict_uq_laplace(laplace_wrapper, front, wrist):
    return laplace_wrapper.predict(front, wrist)


# ---------------------------------------------------------------------------
# UQ Inference
# ---------------------------------------------------------------------------

def enable_mc_dropout(model):
    """Enable dropout layers only, keeping everything else in eval mode."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def mc_dropout_predict(model, front, wrist, T=50):
    enable_mc_dropout(model)
    logits = []
    with torch.no_grad():
        for _ in range(T):
            logits.append(model(front, wrist))
    return torch.stack(logits, dim=0)  # (T, B)


def predict(model, front_bytes, wrist_bytes, mode="mc_dropout", device="cuda", T=50,
            logit_variance=False):
    front_img = decompress_image(front_bytes)
    wrist_img = decompress_image(wrist_bytes)
    front = _img_transform(front_img).unsqueeze(0).to(device)
    wrist = _img_transform(wrist_img).unsqueeze(0).to(device)

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
            all_logits = model(front, wrist)

    elif mode == "combined":
        logit_list = []
        for member in model.members:
            member_logits = mc_dropout_predict(member, front, wrist, T=T)
            logit_list.append(member_logits)
        all_logits = torch.cat(logit_list, dim=0)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    logits_flat = all_logits.squeeze(-1).cpu().numpy()
    probs_flat = 1.0 / (1.0 + np.exp(-logits_flat))
    mean_prob = float(probs_flat.mean())
    variance = float(logits_flat.var() if logit_variance else probs_flat.var())
    p = np.clip(mean_prob, 1e-8, 1 - 1e-8)
    entropy = float(-p * np.log(p) - (1 - p) * np.log(1 - p))
    return {"prob": mean_prob, "variance": variance, "entropy": entropy}


def batch_predict_uq(model, front, wrist, mode="mc_dropout", T=50, logit_variance=False):
    """
    Batch UQ prediction on already-prepared tensors (B, C, H, W).
    Returns: mean_probs (B,), variances (B,), entropies (B,)
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
            all_logits = model(front, wrist)

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
    p = np.clip(mean_probs, 1e-8, 1 - 1e-8)
    entropies = -p * np.log(p) - (1 - p) * np.log(1 - p)
    return mean_probs, variances, entropies


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_ensemble_parallel(models, train_loader, val_loader, args, freeze_backbone=True):
    """Train all ensemble members with a shared DINOv2 backbone."""
    K = len(models)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [m.to(device) for m in models]

    shared_front_enc = models[0].front_encoder
    shared_wrist_enc = models[0].wrist_encoder

    head_params_list = [list(m.head.parameters()) for m in models]
    head_optimizers = [
        torch.optim.Adam(params, lr=args.lr, weight_decay=1e-5)
        for params in head_params_list
    ]
    head_schedulers = [
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        for opt in head_optimizers
    ]

    backbone_params = None
    backbone_optimizer = None
    backbone_scheduler = None
    if not freeze_backbone:
        backbone_params = (list(shared_front_enc.parameters()) +
                           list(shared_wrist_enc.parameters()))
        backbone_optimizer = torch.optim.Adam(
            backbone_params, lr=args.lr * 0.1, weight_decay=1e-5)
        backbone_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            backbone_optimizer, T_max=args.epochs)

    best_val_f1 = [-1.0] * K
    best_head_states = [None] * K

    for epoch in range(args.epochs):
        if freeze_backbone:
            shared_front_enc.eval()
            shared_wrist_enc.eval()
        else:
            shared_front_enc.train()
            shared_wrist_enc.train()
        for m in models:
            m.head.train()

        train_losses = [0.0] * K
        train_correct = [0] * K
        train_total = 0

        for front, wrist, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            front, wrist, labels = front.to(device), wrist.to(device), labels.to(device)
            train_total += labels.size(0)

            if freeze_backbone:
                with torch.no_grad():
                    f_feat = shared_front_enc(front)
                    w_feat = shared_wrist_enc(wrist)
                backbone_feat = torch.cat([f_feat, w_feat], dim=1)
            else:
                f_feat = shared_front_enc(front)
                w_feat = shared_wrist_enc(wrist)
                backbone_feat = torch.cat([f_feat, w_feat], dim=1)

            for opt in head_optimizers:
                opt.zero_grad()
            if backbone_optimizer:
                backbone_optimizer.zero_grad()

            total_loss = 0
            for i in range(K):
                logits = models[i].head(backbone_feat).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, labels)
                total_loss = total_loss + loss
                train_losses[i] += loss.item() * labels.size(0)
                train_correct[i] += ((logits.detach() > 0).float() == labels).sum().item()

            total_loss.backward()

            for i in range(K):
                nn.utils.clip_grad_norm_(head_params_list[i], 5.0)
                head_optimizers[i].step()

            if backbone_optimizer:
                nn.utils.clip_grad_norm_(backbone_params, 5.0)
                backbone_optimizer.step()

        for s in head_schedulers:
            s.step()
        if backbone_scheduler:
            backbone_scheduler.step()

        # Val
        shared_front_enc.eval()
        shared_wrist_enc.eval()
        for m in models:
            m.head.eval()

        val_losses = [0.0] * K
        val_preds_all = [[] for _ in range(K)]
        val_labels_all = []
        val_total = 0

        with torch.no_grad():
            for front, wrist, labels in val_loader:
                front, wrist, labels = front.to(device), wrist.to(device), labels.to(device)
                val_total += labels.size(0)
                val_labels_all.extend(labels.cpu().numpy())

                f_feat = shared_front_enc(front)
                w_feat = shared_wrist_enc(wrist)
                backbone_feat = torch.cat([f_feat, w_feat], dim=1)

                for i in range(K):
                    logits = models[i].head(backbone_feat).squeeze(-1)
                    loss = F.binary_cross_entropy_with_logits(logits, labels)
                    val_losses[i] += loss.item() * labels.size(0)
                    val_preds_all[i].extend((logits > 0).float().cpu().numpy())

        val_labels_np = np.array(val_labels_all)
        for i in range(K):
            tl = train_losses[i] / train_total
            ta = train_correct[i] / train_total
            vl = val_losses[i] / val_total
            vp = np.array(val_preds_all[i])
            va = accuracy_score(val_labels_np, vp)
            vprec = precision_score(val_labels_np, vp, zero_division=0)
            vrec = recall_score(val_labels_np, vp, zero_division=0)
            vf1 = f1_score(val_labels_np, vp, zero_division=0)

            print(
                f"  [M{i}] Epoch {epoch+1}/{args.epochs} | "
                f"TrL: {tl:.4f} TrA: {ta:.4f} | "
                f"VL: {vl:.4f} VA: {va:.4f} P: {vprec:.4f} "
                f"R: {vrec:.4f} F1: {vf1:.4f}"
            )

            if vf1 > best_val_f1[i]:
                best_val_f1[i] = vf1
                best_head_states[i] = {
                    k: v.cpu().clone() for k, v in models[i].head.state_dict().items()
                }

    for i in range(K):
        if best_head_states[i] is not None:
            models[i].head.load_state_dict(best_head_states[i])

    front_sd = shared_front_enc.state_dict()
    wrist_sd = shared_wrist_enc.state_dict()
    for i in range(1, K):
        models[i].front_encoder.load_state_dict(front_sd)
        models[i].wrist_encoder.load_state_dict(wrist_sd)

    return models, best_val_f1


def split_episodes_train_val(episodes, val_frac=0.1, seed=42):
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(episodes))
    n_val = max(1, int(len(episodes) * val_frac))
    val_idx = set(indices[:n_val].tolist())
    train_eps = [ep for i, ep in enumerate(episodes) if i not in val_idx]
    val_eps = [ep for i, ep in enumerate(episodes) if i in val_idx]
    return train_eps, val_eps


def run_train(args):
    print("Loading training data...")
    episodes = load_episodes(TRAIN_FILES)
    print(f"Total episodes: {len(episodes)}")

    train_eps, val_eps = split_episodes_train_val(episodes, val_frac=0.1, seed=42)
    print(f"Train episodes: {len(train_eps)}, Val episodes: {len(val_eps)}")

    train_dataset = FailureDataset(train_eps)
    val_dataset = FailureDataset(val_eps)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_labels = train_dataset.get_labels()
    n_pos = sum(1 for lb in train_labels if lb > 0.5)
    n_neg = len(train_labels) - n_pos
    print(f"Train class distribution: success={n_neg}, failure={n_pos}")

    sampler = make_weighted_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint dir: {ckpt_dir}")

    freeze_backbone = not args.finetune_backbone

    members = []
    for i in range(args.ensemble_size):
        seed = 42 + i * 1000
        model = FailureClassifierDINOv2(
            dropout_rate=args.dropout_rate,
            freeze_backbone=freeze_backbone,
        )
        init_head_weights(model.head, seed=seed)
        members.append(model)

    total_params = sum(p.numel() for p in members[0].parameters())
    trainable_params = sum(p.numel() for p in members[0].parameters() if p.requires_grad)
    print(f"Backbone: DINOv2 ViT-S/14 (dinov2_vits14, embed_dim=384)")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} (head only)")
    print(f"Frozen parameters: {total_params - trainable_params:,} (DINOv2 backbone)")
    print(f"\nTraining {args.ensemble_size} member(s) in parallel on identical batches...")

    ensemble_members, best_f1s = train_ensemble_parallel(
        members, train_loader, val_loader, args,
        freeze_backbone=freeze_backbone,
    )

    for i, (model, best_f1) in enumerate(zip(ensemble_members, best_f1s)):
        print(f"  Member {i}: best val F1 = {best_f1:.4f}")
        member_path = ckpt_dir / f"member_{i}.pt"
        torch.save(model.state_dict(), member_path)
        print(f"  Saved: {member_path}")

    ensemble = FailureClassifierDINOv2Ensemble(
        ensemble_size=args.ensemble_size,
        dropout_rate=args.dropout_rate,
        freeze_backbone=freeze_backbone,
    )
    for i, member in enumerate(ensemble_members):
        ensemble.members[i].load_state_dict(member.state_dict())

    ensemble_path = ckpt_dir / "checkpoint.pt"
    torch.save({
        "ensemble_size": args.ensemble_size,
        "state_dict": ensemble.state_dict(),
        "args": vars(args),
        "backbone": "dinov2_vits14",
    }, ensemble_path)
    print(f"\nSaved checkpoint: {ensemble_path}")


def run_fit_laplace(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ensemble_size = ckpt["ensemble_size"]
    dropout_rate = ckpt.get("args", {}).get("dropout_rate", 0.2)
    freeze_backbone = not ckpt.get("args", {}).get("finetune_backbone", False)

    ensemble = FailureClassifierDINOv2Ensemble(
        ensemble_size=ensemble_size, dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone,
    )
    ensemble.load_state_dict(ckpt["state_dict"])
    model = ensemble.members[0].to(device)
    print(f"Using member 0 (dropout={dropout_rate})")

    print("Loading training data for Hessian computation...")
    episodes = load_episodes(TRAIN_FILES)
    train_eps, _ = split_episodes_train_val(episodes, val_frac=0.1, seed=42)
    train_dataset = FailureDataset(train_eps)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    print(f"Training samples for Hessian: {len(train_dataset)}")

    print(f"\nFitting Laplace (prior_precision={args.prior_precision})...")
    laplace = LaplaceWrapper(model, prior_precision=args.prior_precision)
    laplace.fit(train_loader, device=device)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    laplace_path = ckpt_dir / "laplace_state.pt"
    laplace.save(laplace_path)

    single_ensemble = FailureClassifierDINOv2Ensemble(
        ensemble_size=1, dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone,
    )
    single_ensemble.members[0].load_state_dict(model.state_dict())
    combined_path = ckpt_dir / "checkpoint.pt"
    torch.save({
        "ensemble_size": 1,
        "state_dict": single_ensemble.state_dict(),
        "args": ckpt.get("args", {}),
        "backbone": "dinov2_vits14",
        "laplace_state": {
            "posterior_cov": laplace.posterior_cov.cpu(),
            "prior_precision": laplace.prior_precision,
        },
    }, combined_path)
    print(f"Saved Laplace checkpoint: {combined_path}")


# ---------------------------------------------------------------------------
# Inspect
# ---------------------------------------------------------------------------

def run_inspect(args):
    all_files = TRAIN_FILES + [EVAL_FILE, "calibration_dataset.pkl"]
    for fname in all_files:
        path = DATA_DIR / fname
        if not path.exists():
            print(f"\n{fname}: NOT FOUND")
            continue

        with open(path, "rb") as f:
            episodes = pickle.load(f)

        n_eps = len(episodes)
        lengths = []
        n_success, n_failure, n_unknown = 0, 0, 0

        for ep in episodes:
            fail_arr = ep["failure"]
            T = len(fail_arr)
            lengths.append(T)
            n_success += int((fail_arr == 0).sum())
            n_failure += int((fail_arr == 1).sum())
            n_unknown += int((fail_arr == -1).sum())

        total_steps = sum(lengths)
        lengths = np.array(lengths)

        print(f"\n{'='*60}")
        print(f"{fname}")
        print(f"{'='*60}")
        print(f"  Episodes: {n_eps}")
        print(f"  Total timesteps: {total_steps}")
        print(f"  Trajectory lengths: min={lengths.min()}, max={lengths.max()}, "
              f"mean={lengths.mean():.1f}")
        print(f"  Raw labels:")
        print(f"    Success (0):  {n_success:>7} ({100*n_success/total_steps:.1f}%)")
        print(f"    Failure (1):  {n_failure:>7} ({100*n_failure/total_steps:.1f}%)")
        print(f"    Unknown (-1): {n_unknown:>7} ({100*n_unknown/total_steps:.1f}%)")
        eff_success = n_success + n_unknown
        eff_failure = n_failure
        print(f"  After mapping (-1 -> success):")
        print(f"    Success: {eff_success:>7} ({100*eff_success/total_steps:.1f}%)")
        print(f"    Failure: {eff_failure:>7} ({100*eff_failure/total_steps:.1f}%)")


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def run_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    ensemble_size = ckpt["ensemble_size"]
    dropout_rate = ckpt.get("args", {}).get("dropout_rate", 0.2)
    freeze_backbone = not ckpt.get("args", {}).get("finetune_backbone", False)
    ensemble = FailureClassifierDINOv2Ensemble(
        ensemble_size=ensemble_size,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone,
    )
    ensemble.load_state_dict(ckpt["state_dict"])
    ensemble = ensemble.to(device)
    print(f"Loaded DINOv2 ensemble with {ensemble_size} members (dropout={dropout_rate})")

    print("Loading labeled data for evaluation (episode-level held-out split)...")
    labeled_episodes = load_episodes(["failure_labeled.pkl"])
    _, eval_episodes = split_episodes_train_val(labeled_episodes, val_frac=0.1, seed=42)
    eval_dataset = FailureDataset(eval_episodes)
    eval_labels = eval_dataset.get_labels()
    n_pos = sum(1 for lb in eval_labels if lb > 0.5)
    print(f"Eval samples: {len(eval_dataset)} (success={len(eval_dataset)-n_pos}, failure={n_pos})")

    eval_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    mode = args.mode
    T = 50

    laplace = None
    if mode == "laplace":
        if "laplace_state" not in ckpt:
            print("ERROR: Checkpoint does not contain Laplace state. Run 'fit_laplace' first.")
            return
        model_single = ensemble.members[0]
        laplace = LaplaceWrapper(model_single, prior_precision=ckpt["laplace_state"]["prior_precision"])
        laplace.posterior_cov = ckpt["laplace_state"]["posterior_cov"].to(device)
        laplace.fitted = True
        print(f"Loaded Laplace state (prior_precision={laplace.prior_precision})")

    logit_var = args.logit_variance
    var_space = "logit" if args.logit_variance else "probability"
    all_probs, all_vars, all_entropies, all_labels = [], [], [], []

    print(f"Running evaluation with mode={mode}, variance_space={var_space}...")
    t0 = time.time()
    for front, wrist, labels in eval_loader:
        front, wrist = front.to(device), wrist.to(device)
        if mode == "laplace":
            probs, variances, entropies = batch_predict_uq_laplace(laplace, front, wrist)
        else:
            with torch.no_grad():
                probs, variances, entropies = batch_predict_uq(
                    ensemble, front, wrist, mode=mode, T=T,
                    logit_variance=logit_var,
                )
        all_probs.extend(probs)
        all_vars.extend(variances)
        all_entropies.extend(entropies)
        all_labels.extend(labels.numpy())

    elapsed = time.time() - t0
    print(f"Evaluation took {elapsed:.1f}s")

    all_probs = np.array(all_probs)
    all_vars = np.array(all_vars)
    all_entropies = np.array(all_entropies)
    all_labels = np.array(all_labels)

    preds = (all_probs > 0.5).astype(float)
    correct_mask = preds == all_labels

    acc = accuracy_score(all_labels, preds)
    prec = precision_score(all_labels, preds, zero_division=0)
    rec = recall_score(all_labels, preds, zero_division=0)
    f1 = f1_score(all_labels, preds, zero_division=0)
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auroc = float("nan")

    print(f"\n{'='*50}")
    print(f"Classification Metrics (mode={mode})")
    print(f"{'='*50}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  AUROC:     {auroc:.4f}")

    print(f"\n{'='*50}")
    print(f"UQ Metrics")
    print(f"{'='*50}")
    print(f"  Overall   | Variance: {all_vars.mean():.6f}  Entropy: {all_entropies.mean():.4f}")
    if correct_mask.sum() > 0:
        print(f"  Correct   | Variance: {all_vars[correct_mask].mean():.6f}  "
              f"Entropy: {all_entropies[correct_mask].mean():.4f}")
    if (~correct_mask).sum() > 0:
        print(f"  Incorrect | Variance: {all_vars[~correct_mask].mean():.6f}  "
              f"Entropy: {all_entropies[~correct_mask].mean():.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Failure Classifier (DINOv2) with UQ")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("inspect", help="Print dataset statistics")

    train_parser = subparsers.add_parser("train", help="Train failure classifier")
    train_parser.add_argument("--checkpoint_dir", type=str,
                              default=str(DEFAULT_CKPT_DIR),
                              help="Directory to save checkpoints")
    train_parser.add_argument("--ensemble_size", type=int, default=5)
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch_size", type=int, default=64)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--dropout_rate", type=float, default=0.2)
    train_parser.add_argument("--finetune_backbone", action="store_true",
                              help="If set, also fine-tune the DINOv2 backbone (not just head)")

    laplace_parser = subparsers.add_parser("fit_laplace",
        help="Fit Laplace approximation on trained baseline model")
    laplace_parser.add_argument("--checkpoint", type=str, required=True)
    laplace_parser.add_argument("--checkpoint_dir", type=str,
                                default=str(DEFAULT_CKPT_DIR / "laplace"))
    laplace_parser.add_argument("--prior_precision", type=float, default=1.0)
    laplace_parser.add_argument("--batch_size", type=int, default=64)

    eval_parser = subparsers.add_parser("eval", help="Evaluate model")
    eval_parser.add_argument("--checkpoint", type=str, required=True)
    eval_parser.add_argument("--mode", type=str, default="ensemble",
                             choices=["baseline", "mc_dropout", "ensemble", "combined", "laplace"])
    eval_parser.add_argument("--batch_size", type=int, default=64)
    eval_parser.add_argument("--logit_variance", action="store_true",
                             help="Compute variance in logit space instead of probability space")

    args = parser.parse_args()

    if args.command == "inspect":
        run_inspect(args)
    elif args.command == "train":
        run_train(args)
    elif args.command == "fit_laplace":
        run_fit_laplace(args)
    elif args.command == "eval":
        run_eval(args)


if __name__ == "__main__":
    main()
