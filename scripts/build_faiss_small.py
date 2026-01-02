from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import faiss

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMB_PATH = PROJECT_ROOT / "data" / "embeddings" / "video_emb_small.npy"
IDS_PATH = PROJECT_ROOT / "data" / "embeddings" / "video_ids_small.json"

OUT_DIR = PROJECT_ROOT / "data" / "index"
OUT_INDEX = OUT_DIR / "video_small.index"
OUT_IDMAP = OUT_DIR / "video_small_idmap.json"


def main() -> None:
    if not EMB_PATH.exists() or not IDS_PATH.exists():
        raise FileNotFoundError("Missing embeddings. Run extract_embeddings_small.py first.")

    X = np.load(EMB_PATH).astype("float32")  # (N, 512)

    with open(IDS_PATH, "r") as f:
        ids = json.load(f)

    if X.shape[0] != len(ids):
        raise RuntimeError(f"Mismatch: embeddings rows={X.shape[0]} but ids={len(ids)}")

    # Safety: ensure vectors are normalized (cosine similarity via dot product)
    faiss.normalize_L2(X)

    d = X.shape[1]
    index = faiss.IndexFlatIP(d)  # Inner Product index
    index.add(X)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(OUT_INDEX))

    with open(OUT_IDMAP, "w") as f:
        json.dump({"video_ids": ids}, f)

    print(f"✅ Built FAISS index: {OUT_INDEX}")
    print(f"   ntotal={index.ntotal}, dim={d}")
    print(f"✅ Saved id map: {OUT_IDMAP}")


if __name__ == "__main__":
    main()
