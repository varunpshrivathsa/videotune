from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import CLIPModel, CLIPProcessor


# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAP_CSV = PROJECT_ROOT / "data" / "processed" / "video_caption_map.csv"

OUT_DIR = PROJECT_ROOT / "data" / "embeddings"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEXT_NPY = OUT_DIR / "text_clip_train7k.npy"
TEXT_IDMAP_JSON = OUT_DIR / "text_clip_train7k_idmap.json"

MODEL_NAME = "openai/clip-vit-base-patch32"

BATCH_TEXT = 256  # adjust if you see GPU OOM, try 128


def clean_caption(x: str) -> str:
    """Basic cleanup so tokenization is consistent."""
    if not isinstance(x, str):
        return ""
    x = x.strip()
    # optional: collapse multiple spaces
    x = " ".join(x.split())
    return x


@torch.inference_mode()
def main() -> None:
    # 1) Load CSV
    df = pd.read_csv(MAP_CSV)

    # Expect columns: video_id, video_path, caption
    assert "video_id" in df.columns, "CSV must contain 'video_id'"
    assert "caption" in df.columns, "CSV must contain 'caption'"

    captions = df["caption"].astype(str).map(clean_caption).tolist()
    video_ids = df["video_id"].astype(str).tolist()

    # 2) Setup device + CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    # use_fast=True removes the warning and is future-proof
    processor = CLIPProcessor.from_pretrained(MODEL_NAME, use_fast=False)

    # 3) Batch encode text -> embeddings
    all_embs: list[np.ndarray] = []
    total = len(captions)

    for start in tqdm(range(0, total, BATCH_TEXT), desc="Encoding text"):
        batch_caps = captions[start : start + BATCH_TEXT]

        # Tokenize text
        inputs = processor(text=batch_caps, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Text features: (B, D)
        feats = model.get_text_features(**inputs)

        # Normalize for cosine similarity / FAISS IP search
        feats = feats / feats.norm(dim=-1, keepdim=True)

        # Move to CPU numpy float32
        feats_np = feats.detach().cpu().numpy().astype("float32")
        all_embs.append(feats_np)

    X = np.concatenate(all_embs, axis=0)

    # 4) Safety check: alignment
    if X.shape[0] != len(video_ids):
        raise RuntimeError(f"Row mismatch: embeddings={X.shape[0]} vs video_ids={len(video_ids)}")

    # 5) Save outputs
    np.save(TEXT_NPY, X)

    with open(TEXT_IDMAP_JSON, "w") as f:
        json.dump(
            {
                "video_ids": video_ids,
                "model": MODEL_NAME,
                "source_csv": str(MAP_CSV),
                "embedding_type": "text",
                "normalized": True,
            },
            f,
            indent=2,
        )

    print("Saved text embeddings:", TEXT_NPY, "shape:", X.shape)
    print("Saved idmap:", TEXT_IDMAP_JSON)


if __name__ == "__main__":
    main()
