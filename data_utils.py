"""
Dataset loading, image transforms, and sampling utilities.
"""

import pickle

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

from config import DATA_DIR, IMAGENET_MEAN, IMAGENET_STD


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def decompress_image(compressed_img):
    """Decompress JPEG bytes to RGB uint8 numpy array (256x256x3)."""
    encoded = np.frombuffer(compressed_img, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    assert bgr.shape[:2] == (256, 256), f"Unexpected shape: {bgr.shape}"
    return bgr[:, :, ::-1].copy()  # BGR -> RGB


# DINOv2 preprocessing: resize to 224, ImageNet-normalize.
_img_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# Episode loading
# ---------------------------------------------------------------------------

def load_episodes(filenames, data_dir=None):
    """Load episodes from pkl files. Returns list of episode dicts."""
    data_dir = data_dir or DATA_DIR
    episodes = []
    for fname in filenames:
        path = data_dir / fname
        if not path.exists():
            print(f"Warning: {path} not found, skipping")
            continue
        with open(path, "rb") as f:
            eps = pickle.load(f)
        print(f"  Loaded {len(eps)} episodes from {fname}")
        episodes.extend(eps)
    return episodes


def split_episodes_train_val(episodes, val_frac=0.1, seed=42):
    """Split episodes into train / val sets (episode-level, deterministic)."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(episodes))
    n_val = max(1, int(len(episodes) * val_frac))
    val_idx = set(indices[:n_val].tolist())
    train_eps = [ep for i, ep in enumerate(episodes) if i not in val_idx]
    val_eps   = [ep for i, ep in enumerate(episodes) if i in val_idx]
    return train_eps, val_eps


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

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
# Visualization helpers
# ---------------------------------------------------------------------------

def sample_frames(episode, n_frames=8):
    """Return *n_frames* evenly-spaced timestep indices for an episode."""
    T = len(episode["failure"])
    if T <= n_frames:
        return list(range(T))
    return [int(i * (T - 1) / (n_frames - 1)) for i in range(n_frames)]
