"""
DINOv2 failure classifier, ensemble wrapper, and Laplace approximation.
"""

import torch
import torch.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# DINOv2 backbone
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


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def _init_head_weights(head, seed=None):
    """Kaiming init for MLP head.  Different seed → different ensemble member."""
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
        front_cam → DINOv2 ViT-S/14 (frozen) → 384-dim  ┐
        wrist_cam → DINOv2 ViT-S/14 (frozen) → 384-dim  ┘ concat → 768-dim
                           │
              Linear(768, 512)  + ReLU + Dropout
              Linear(512, 256)  + ReLU + Dropout
              Linear(256, 128)  + ReLU + Dropout
              Linear(128, 128)  + ReLU + Dropout
              Linear(128, 1)    → logit
    """

    def __init__(self, dropout_rate=0.2, freeze_backbone=True):
        super().__init__()
        self.front_encoder = DINOv2Encoder(freeze=freeze_backbone)
        self.wrist_encoder = DINOv2Encoder(freeze=freeze_backbone)
        embed_dim = self.front_encoder.embed_dim  # 384
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),  # 768 → 512
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
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


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

class FailureClassifierDINOv2Ensemble(nn.Module):
    """Ensemble of K FailureClassifierDINOv2 instances with shared backbone."""

    def __init__(self, ensemble_size=5, dropout_rate=0.2, freeze_backbone=True):
        super().__init__()
        self.members = nn.ModuleList()
        for i in range(ensemble_size):
            member = FailureClassifierDINOv2(dropout_rate, freeze_backbone)
            _init_head_weights(member.head, seed=i * 9999 + 7)
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

        D = self.model.last_layer.in_features  # 128
        H = torch.zeros(D + 1, D + 1, device=device)
        n_samples = 0

        with torch.no_grad():
            for front, wrist, _ in dataloader:
                front, wrist = front.to(device), wrist.to(device)
                B = front.size(0)
                phi = self.model.get_features(front, wrist)
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
        """
        Returns:
            probs:      (B,) numpy – probit-adjusted probabilities
            logit_vars: (B,) numpy – predictive logit variance
            entropies:  (B,) numpy – binary entropy
        """
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

    def load(self, path, device="cuda"):
        state = torch.load(path, map_location=device, weights_only=False)
        self.posterior_cov = state["posterior_cov"].to(device)
        self.prior_precision = state["prior_precision"]
        self.fitted = True


# ---------------------------------------------------------------------------
# Checkpoint loading helpers
# ---------------------------------------------------------------------------

def load_classifier(checkpoint_path, device, mode=None):
    """
    Load a DINOv2 ensemble checkpoint.

    Returns:
        model:   FailureClassifierDINOv2Ensemble on *device*
        laplace: LaplaceWrapper (or None if mode != "laplace")
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ensemble_size = ckpt["ensemble_size"]
    train_dropout = ckpt.get("args", {}).get("dropout_rate", 0.2)
    freeze_backbone = not ckpt.get("args", {}).get("finetune_backbone", False)

    print(f"Loading classifier from {checkpoint_path}")
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

    print(f"Loaded DINOv2 classifier: K={ensemble_size}, dropout={train_dropout}, mode={mode}")
    return model, laplace
