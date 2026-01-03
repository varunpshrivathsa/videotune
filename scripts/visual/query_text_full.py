from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import faiss
import torch
import clip


PROJECT_ROOT = Path(__file__).resolve().parents[1]

META_CSV = PROJECT_ROOT / "data" / "processed" / "metadata.csv"
INDEX_PATH = PROJECT_ROOT / "data" / "index" / "video_full.index"
IDMAP_PATH = PROJECT_ROOT / "data" / "index" / "video_full_idmap.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B/32"


@torch.no_grad()
def embed_text(query: str) -> np.ndarray:
    """
    Text -> (512,) CLIP text embedding.
    Normalize so dot-product == cosine similarity.
    """
    model, _ = clip.load(MODEL_NAME, device=DEVICE)
    model.eval()

    tokens = clip.tokenize([query], truncate=True).to(DEVICE)  # (1, 77)
    feat = model.encode_text(tokens).float()                   # (1, 512)
    feat = feat / feat.norm(dim=-1, keepdim=True)              # unit norm
    return feat[0].cpu().numpy().astype("float32")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True, help="Text query")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    # Captions for nice display
    df = pd.read_csv(META_CSV)
    caption_map = dict(zip(df["video_id"].astype(str), df["caption"].astype(str)))

    # Load FAISS index + id map
    index = faiss.read_index(str(INDEX_PATH))
    ids = json.loads(Path(IDMAP_PATH).read_text())["video_ids"]

    # Build query vector from text
    q = embed_text(args.q).reshape(1, -1)
    faiss.normalize_L2(q)  # extra safety

    scores, idx = index.search(q, args.topk)

    print(f"\nQuery text: {args.q}\n")
    print("Top matches:")
    for rank, (i, s) in enumerate(zip(idx[0], scores[0]), start=1):
        i = int(i)
        vid = ids[i]
        print(f"{rank:02d}. {vid}  score={float(s):.4f}")
        print(f"    caption: {caption_map.get(vid, 'N/A')}")


if __name__ == "__main__":
    main()
