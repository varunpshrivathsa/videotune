from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import faiss

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INDEX_PATH = PROJECT_ROOT / "data" / "index" / "video_small.index"
IDMAP_PATH = PROJECT_ROOT / "data" / "index" / "video_small_idmap.json"
EMB_PATH = PROJECT_ROOT / "data" / "embeddings" / "video_emb_small.npy"


def main() -> None:
    # Load index + id map
    index = faiss.read_index(str(INDEX_PATH))
    with open(IDMAP_PATH, "r") as f:
        ids = json.load(f)["video_ids"]

    # Load embeddings (so we can pick a query vector by row index)
    X = np.load(EMB_PATH).astype("float32")
    faiss.normalize_L2(X)

    query_row = 0          # video0 (because we took first 50)
    topk = 6

    q = X[query_row].reshape(1, -1)
    scores, idx = index.search(q, topk)

    print(f"Query video: {ids[query_row]}")
    print("Top matches (excluding itself):")
    shown = 0
    for i, s in zip(idx[0], scores[0]):
        i = int(i)
        if i == query_row:
            continue
        shown += 1
        print(f"{shown:02d}. {ids[i]}  score={float(s):.4f}")
        if shown == 5:
            break


if __name__ == "__main__":
    main()
