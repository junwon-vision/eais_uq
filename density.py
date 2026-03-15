"""
Flow-matching density estimator on DINOv2 features.

The ConditionalUnet1D and its building blocks are inlined here so that
this package has no dependency on the external ``diffusion_policy`` tree.
"""

import math
from typing import Union

import einops
import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from data_utils import _img_transform, decompress_image
from models import DINOv2Encoder


# ===================================================================
# Inlined UNet components (from diffusion_policy)
# ===================================================================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Conv1dBlock(nn.Module):
    """Conv1d → GroupNorm → Mish."""

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size,
                      padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class ConditionalResidualBlock1D(nn.Module):
    """Residual block with FiLM conditioning."""

    def __init__(self, in_channels, out_channels, cond_dim,
                 kernel_size=3, n_groups=8, cond_predict_scale=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        cond_channels = out_channels * 2 if cond_predict_scale else out_channels
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.residual_conv = (nn.Conv1d(in_channels, out_channels, 1)
                              if in_channels != out_channels else nn.Identity())

    def forward(self, x, cond):
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    """1-D UNet with diffusion-step conditioning for flow matching."""

    def __init__(self, input_dim, local_cond_dim=None, global_cond_dim=None,
                 diffusion_step_embed_dim=256, down_dims=(256, 512, 1024),
                 kernel_size=3, n_groups=8, cond_predict_scale=False):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                nn.Identity(),
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out * 2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                nn.Identity(),
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = None
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

    def forward(self, sample: torch.Tensor,
                timestep: Union[torch.Tensor, float, int],
                local_cond=None, global_cond=None, **kwargs):
        """
        sample:   (B, T, input_dim)
        timestep: (B,) or scalar
        output:   (B, T, input_dim)
        """
        sample = einops.rearrange(sample, 'b h t -> b t h')

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long,
                                     device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        x = sample
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        x = einops.rearrange(x, 'b t h -> b h t')
        return x


# ===================================================================
# Flow Matching Density Estimator
# ===================================================================

def _build_unet(input_dim):
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

    Training:  learns v(x, t) mapping data → Gaussian.
    Inference: z = x + v(x, 0);  logpZO = ||z||²  (higher = more OOD).
    """

    def __init__(self, input_dim, device="cuda"):
        super().__init__()
        self.input_dim = input_dim
        self.net = _build_unet(input_dim)
        self.time_scale = 100

    def train_step(self, features):
        """One flow-matching training step. Returns scalar loss."""
        self.net.train()
        device = features.device
        B = features.shape[0]

        x0 = features.reshape(B, 1, self.input_dim)
        x1 = torch.randn_like(x0)
        v_true = x1 - x0

        t = torch.rand(B, device=device)
        t_expand = t.view(B, 1, 1)
        x_t = x0 + t_expand * v_true

        v_hat = self.net(x_t, (t * self.time_scale).long())
        loss = (v_hat - v_true).pow(2).mean()
        return loss

    @torch.no_grad()
    def compute_logpZO(self, features):
        """
        Compute log-density proxy.

        Returns:
            logpZO: (B,) tensor — higher = lower density = more uncertain
        """
        self.net.eval()
        B = features.shape[0]

        x = features.reshape(B, 1, self.input_dim)
        timesteps = torch.zeros(B, device=features.device, dtype=torch.long)
        v_hat = self.net(x, timesteps)

        z = x + v_hat
        logpZO = z.reshape(B, -1).pow(2).sum(dim=-1)
        return logpZO


# ===================================================================
# Loading & Inference API
# ===================================================================

def load_density_model(checkpoint_path, device):
    """Load trained density model. Returns (model, dino_encoder)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    input_dim = ckpt["input_dim"]
    model = FlowMatchingDensity(input_dim, device=device).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    encoder = DINOv2Encoder(freeze=True).to(device)
    encoder.eval()

    print(f"Loaded density model (input_dim={input_dim})")
    return model, encoder


@torch.no_grad()
def predict_density_episode(model, encoder, ep, device, batch_size=32):
    """
    Run density estimation on all timesteps of an episode.

    Returns:
        logpZO: (T,) numpy array — higher = more OOD / uncertain
    """
    T = len(ep["failure"])
    all_logpZO = []

    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        fronts, wrists = [], []
        for t in range(start, end):
            f_img = ep["front_cam"][t]
            w_img = ep["wrist_cam"][t]
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
        combined = torch.cat([f_feat, w_feat], dim=1)

        logpZO = model.compute_logpZO(combined).cpu().numpy()
        all_logpZO.extend(logpZO)

    return np.array(all_logpZO)


@torch.no_grad()
def extract_features_from_episodes(episodes, device, batch_size=64):
    """
    Extract DINOv2 features for all timesteps across episodes.

    Returns:
        features:      (N, 768) tensor
        labels:        (N,) numpy array
        ep_boundaries: list of (start, end) per episode
    """
    encoder = DINOv2Encoder(freeze=True).to(device)
    encoder.eval()

    all_features, all_labels = [], []
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
            f_feat = encoder(fronts)
            w_feat = encoder(wrists)
            combined = torch.cat([f_feat, w_feat], dim=1)
            ep_feats.append(combined.cpu())

        ep_feats = torch.cat(ep_feats, dim=0)
        labels = np.array(failure, dtype=np.float32)
        labels[labels < 0] = 0.0

        all_features.append(ep_feats)
        all_labels.append(labels)
        ep_boundaries.append((idx, idx + T))
        idx += T

    features = torch.cat(all_features, dim=0)
    labels = np.concatenate(all_labels)
    return features, labels, ep_boundaries
