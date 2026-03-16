# Uncertainty Quantification & Conformal Calibration for Robotic Failure Prediction

Failure prediction for a robotic Jenga block-stacking task using a frozen **DINOv2 ViT-S/14** backbone. Four UQ methods are implemented and calibrated with conformal prediction to provide coverage guarantees at trajectory level.

---

## Quickstart: Tutorial Notebook

The main content is `uq.ipynb` — a self-contained walkthrough that requires **no local setup or model training**. Pre-trained checkpoints are downloaded automatically in the first cell.

**Run in Google Colab (recommended):** open `uq.ipynb` and execute the first cell, which clones the repo and downloads all checkpoints:

(If you want to run the notebook locally, you can skip this step and follow the local setup instructions below.)
```python
!git clone https://github.com/junwon-vision/eais_uq.git

import gdown
gdown.download(id="1d9W43ejh3Fv98tmnwhCBqZVYms86U_0U", output="model_checkpoints.zip", quiet=False)
!unzip -q -o model_checkpoints.zip -d .
```

The notebook then guides you through:

1. Dataset inspection (Jenga trajectories, front + wrist cameras)
2. Baseline failure classifier inference
3. All four UQ methods — formulation, implementation, and per-episode visualization
4. OOD evaluation on out-of-distribution images and trajectories
5. Conformal calibration — image-level and trajectory-level thresholds with coverage guarantees
6. VLM-based failure classifier (Qwen2.5-VL, zero-shot) with conformal calibration

---

## Methods

| Method | Key idea |
|---|---|
| **Baseline entropy** | Single deterministic forward pass; binary entropy of P(fail) |
| **MC Dropout** | Dropout kept on at test time; variance over T=50 stochastic passes |
| **Deep Ensemble** | K=10 independently trained heads; variance over member predictions |
| **Flow-matching density** | ConditionalUnet1D maps DINOv2 features → Gaussian; logpZO as OOD score |

All methods are calibrated with **split conformal prediction** at both image-level and trajectory-level (max over timesteps), yielding a threshold τ with the marginal guarantee P(u < τ) ≥ 1 − α.

---

## Repository Structure

```
final_hw/
├── uq.ipynb                              # Main tutorial notebook (start here)
├── train_failure_classifier_dinov2.py    # (Optional) Train baseline / MC Dropout / Ensemble
├── train_density_dinov2.py               # (Optional) Train flow-matching density estimator
├── visualize_prediction_video_dinov2_uq.py   # UQ-only overlay videos
├── visualize_prediction_video_density.py     # P(fail) + density UQ overlay videos
│
├── models.py          # DINOv2Encoder, FailureClassifierDINOv2, Ensemble, LaplaceWrapper
├── uq_inference.py    # mc_dropout_predict, run_inference_episode, predict_density_episode
├── conformal.py       # calibrate_image_id, calibrate_traj_id
├── density.py         # Flow-matching model definition and inference
├── data_utils.py      # Dataset loading, image decompression, transform
├── visualization.py   # Plot helpers (trajectory overlays, threshold plots, videos)
├── vlm_classifier.py  # Qwen2.5-VL zero-shot failure classifier
├── config.py          # Shared constants (paths, alpha, device)
│
├── checkpoints/       # Downloaded automatically by the notebook (see above)
│   ├── failure_classifier_dinov2/{baseline,mc_dropout,ensemble,laplace}/
│   └── density_dinov2/checkpoint.pt
│
└── install.sh         # Conda environment setup (local use only)
```

---

## Local Setup

Only needed if running scripts outside the notebook:

```bash
bash install.sh
conda activate uq_safety
```

Requires CUDA 12.8. See `install.sh` for the full dependency list.

---

## Training (Optional)

The notebook uses pre-trained checkpoints downloaded via gdown. If you want to train from scratch:

**Failure classifier** (baseline / MC Dropout / Ensemble):

```bash
# Baseline (deterministic)
python train_failure_classifier_dinov2.py train --ensemble_size 1 --dropout_rate 0.0

# MC Dropout
python train_failure_classifier_dinov2.py train --ensemble_size 1 --dropout_rate 0.2

# Deep Ensemble (K=10)
python train_failure_classifier_dinov2.py train --ensemble_size 10 --dropout_rate 0.0
```

**Density estimator** (flow matching on DINOv2 features):

```bash
python train_density_dinov2.py train \
    --checkpoint_dir checkpoints/density_dinov2 --epochs 50
```

---

## Visualization (Optional)

Generate per-episode videos with UQ score overlays and conformal threshold line:

```bash
# UQ-only overlay (colored border: green=OK / red=FAILURE)
python visualize_prediction_video_dinov2_uq.py \
    --checkpoint checkpoints/failure_classifier_dinov2/ensemble/checkpoint.pt \
    --mode ensemble --calibration calibration.json --all

# P(fail) + density UQ overlay
python visualize_prediction_video_density.py \
    --classifier_checkpoint checkpoints/failure_classifier_dinov2/ensemble/checkpoint.pt \
    --density_checkpoint checkpoints/density_dinov2/checkpoint.pt \
    --mode ensemble --all
```

---

## Dataset

Episodes are stored as `.pkl` files with keys `front_cam`, `wrist_cam`, `failure` (per-timestep binary label). Downloaded as part of the checkpoint archive.

```
datasets/dreamer_fixed/
├── failure_labeled.pkl   # Training set
├── failure_eval.pkl      # Evaluation set
└── success_only.pkl      # Success-only trajectories (for density training)
```

---

## References
- Ren et al. (2023) — KnowNo: [arxiv:2307.01928](https://arxiv.org/abs/2307.01928)
- Seo et al. (2025) — UniSafe: [arxiv:2505.00779](https://arxiv.org/abs/2505.00779), [code](https://github.com/CMU-IntentLab/UNISafe)
- Xu et al. (2025) — FAIL-Detect: [https://github.com/CXU-TRI/FAIL-Detect](https://github.com/CXU-TRI/FAIL-Detect)
