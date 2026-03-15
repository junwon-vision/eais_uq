"""
VLM-based failure classifier using Qwen2.5-VL.

Uses the token probability of answering "yes" / "no" to a failure-detection
prompt as a calibratable UQ score — no fine-tuning required.
"""

import numpy as np
import torch
from PIL import Image

from data_utils import decompress_image


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_vlm(model_name="Qwen/Qwen2.5-VL-3B-Instruct", device="cuda"):
    """
    Load a Qwen2.5-VL model and processor from HuggingFace.

    Returns (model, processor).
    """
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device,
    )
    processor = AutoProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_failure_prompt():
    """Return the text prompt for Jenga failure detection."""
    return (
        "You are a robotic manipulation safety monitor. "
        "The two images show the same moment from two camera views "
        "(front and wrist) of a robot performing a Jenga block stacking task. "
        "Is the Jenga tower toppling, or are blocks falling down? "
        "Answer with a single word: yes or no."
    )


# ---------------------------------------------------------------------------
# Single prediction
# ---------------------------------------------------------------------------

def _make_pil(img):
    """Convert to PIL Image from raw numpy or JPEG bytes."""
    if isinstance(img, (bytes, np.bytes_)):
        img = decompress_image(img)
    return Image.fromarray(img)


def predict_with_vlm(model, processor, front_img, wrist_img):
    """
    Predict failure probability from two camera images.

    Returns dict: {"prob": float, "entropy": float}
        prob    — P(failure) derived from yes/no token logits
        entropy — binary entropy of the prediction
    """
    front_pil = _make_pil(front_img)
    wrist_pil = _make_pil(wrist_img)
    prompt = build_failure_prompt()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": front_pil},
                {"type": "image", "image": wrist_pil},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    inputs = processor(
        text=[text], images=[front_pil, wrist_pil],
        padding=True, return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True,
        )

    # Extract logits for the first generated token
    logits = outputs.scores[0][0]  # (vocab_size,)

    # Find token ids for "yes" and "no"
    yes_ids = processor.tokenizer.encode("yes", add_special_tokens=False)
    no_ids  = processor.tokenizer.encode("no",  add_special_tokens=False)
    # Also try capitalized variants
    yes_ids_cap = processor.tokenizer.encode("Yes", add_special_tokens=False)
    no_ids_cap  = processor.tokenizer.encode("No",  add_special_tokens=False)

    yes_logit = max(logits[yes_ids[0]].item(),
                    logits[yes_ids_cap[0]].item() if yes_ids_cap else -1e9)
    no_logit  = max(logits[no_ids[0]].item(),
                    logits[no_ids_cap[0]].item() if no_ids_cap else -1e9)

    # Softmax over [yes, no]
    pair = torch.tensor([yes_logit, no_logit], dtype=torch.float32)
    probs = torch.softmax(pair, dim=0)
    p_fail = probs[0].item()

    p = np.clip(p_fail, 1e-8, 1 - 1e-8)
    entropy = float(-p * np.log(p) - (1 - p) * np.log(1 - p))

    return {"prob": p_fail, "entropy": entropy}


# ---------------------------------------------------------------------------
# Episode-level prediction
# ---------------------------------------------------------------------------

def predict_vlm_episode(model, processor, ep, device, stride=5):
    """
    Run VLM prediction on an episode with a given stride.

    For non-strided timesteps, the prediction is linearly interpolated
    from the nearest evaluated neighbours.

    Returns:
        probs:     (T,) numpy array
        entropies: (T,) numpy array
    """
    T = len(ep["failure"])
    eval_indices = list(range(0, T, stride))
    if eval_indices[-1] != T - 1:
        eval_indices.append(T - 1)

    eval_probs = {}
    eval_ents = {}
    for t in eval_indices:
        result = predict_with_vlm(model, processor,
                                  ep["front_cam"][t], ep["wrist_cam"][t])
        eval_probs[t] = result["prob"]
        eval_ents[t] = result["entropy"]

    # Interpolate
    probs = np.zeros(T)
    entropies = np.zeros(T)
    for t in range(T):
        if t in eval_probs:
            probs[t] = eval_probs[t]
            entropies[t] = eval_ents[t]
        else:
            # Find surrounding evaluated indices
            left = max(i for i in eval_indices if i <= t)
            right = min(i for i in eval_indices if i >= t)
            if left == right:
                probs[t] = eval_probs[left]
                entropies[t] = eval_ents[left]
            else:
                w = (t - left) / (right - left)
                probs[t] = eval_probs[left] * (1 - w) + eval_probs[right] * w
                entropies[t] = eval_ents[left] * (1 - w) + eval_ents[right] * w

    return probs, entropies
