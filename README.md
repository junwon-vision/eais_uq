# Uncertainty Quantification & Conformal Calibration for Robotic Failure Prediction

Failure prediction for a robotic Jenga block-stacking task using a frozen **DINOv2 ViT-S/14** backbone. Four UQ methods are implemented and calibrated with conformal prediction to provide coverage guarantees at trajectory level.

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
├── uq.ipynb                              # Self-contained tutorial notebook (see below)
├── train_failure_classifier_dinov2.py    # Train baseline / MC Dropout / Ensemble classifier
├── train_density_dinov2.py               # Train flow-matching density estimator
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
├── checkpoints/
│   ├── failure_classifier_dinov2/{baseline,mc_dropout,ensemble,laplace}/
│   └── density_dinov2/checkpoint.pt
│
└── install.sh         # Conda environment setup (see below)
```

---

## Setup

```bash
bash install.sh
conda activate uq_safety
```

Requires CUDA 12.8. See `install.sh` for the full dependency list.

---

## Training

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

## Visualization

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

## Tutorial Notebook

`uq.ipynb` is a self-contained walkthrough covering:

1. Dataset inspection (Jenga trajectories, front + wrist cameras)
2. Baseline failure classifier inference
3. All four UQ methods — implementation and per-episode visualization
4. OOD evaluation on out-of-distribution images and trajectories
5. Conformal calibration — image-level and trajectory-level thresholds
6. VLM-based failure classifier (Qwen2.5-VL, zero-shot) with conformal calibration

```bash
jupyter notebook uq.ipynb
```

---

## Dataset

Episodes are stored as `.pkl` files with keys `front_cam`, `wrist_cam`, `failure` (per-timestep binary label).

```
datasets/dreamer_fixed/
├── failure_labeled.pkl   # Training set
├── failure_eval.pkl      # Evaluation set
└── success_only.pkl      # Success-only trajectories (for density training)
```

---

## References
- Ren et al. (2023) — KnowNo: [arxiv:2307.01928](https://arxiv.org/abs/2307.01928)
- Seo et al. (2025) — UniSafe: [arxiv:2505.00779](https://arxiv.org/abs/2505.00779)
