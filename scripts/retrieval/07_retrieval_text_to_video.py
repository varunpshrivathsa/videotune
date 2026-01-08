from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import faiss
import torch
from transformers import CLIPModel, CLIPProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IND_DIR = PROJECT_ROOT / "data" / "faiss"
PROC_DIR = PROJECT_ROOT / "data" / "processed"

INDEX_PATH = IND_DIR / "video_clip_train7k.index"
INDEX_IDMAP = IND_DIR / "video_clip_train7k.index_idmap.json"
MAP_CSV = PROC_DIR / "video_caption_map.csv"

MODEL_NAME = "openai/clip-vit-base-patch32"

@torch.inference_mode()
def encode_text(query: str, device: str) -> np.ndarray:
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    # Use slow processor to avoid torchvision dependency
    processor = CLIPProcessor.from_pretrained(MODEL_NAME, use_fast=False)

    inputs = processor(text=[query], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    feat = model.get_text_features(**inputs)  # (1, D)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.detach().cpu().numpy().astype("float32")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--query", type=str, required=True, help="Text query")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Load FAISS index + idmap
    index = faiss.read_index(str(INDEX_PATH))
    with open(INDEX_IDMAP, "r") as f:
        idmeta = json.load(f)
    video_ids = idmeta["video_ids"]

    # Load caption map (for nice printing)
    df = pd.read_csv(MAP_CSV)
    cap_map = dict(zip(df["video_id"].astype(str), df["caption"].astype(str)))
    path_map = dict(zip(df["video_id"].astype(str), df["video_path"].astype(str))) if "video_path" in df.columns else {}

    # Encode query -> search
    q = encode_text(args.query, device=device)  # (1, D)
    scores, idxs = index.search(q, args.topk)

    print("\nQUERY:", args.query)
    for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), start=1):
        idx = int(idx)
        vid = video_ids[idx]
        cap = cap_map.get(vid, "")
        vpath = path_map.get(vid, "")
        print(f"{rank:02d}. score={float(score):.4f}  video_id={vid}")
        if cap:
            print(f"    caption: {cap}")
        if vpath:
            print(f"    path: {vpath}")


if __name__ == "__main__":
    main()