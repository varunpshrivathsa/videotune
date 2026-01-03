from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import faiss

PROJECT_ROOT = Path(__file__).resolve().parents[2]

META_CSV = PROJECT_ROOT / "data" / "processed" / "metadata.csv"
INDEX_PATH = PROJECT_ROOT / "data" / "index" / "video_small.index"
IDMAP_PATH = PROJECT_ROOT / "data" / "index" / "video_small_idmap.json"
EMB_PATH = PROJECT_ROOT / "data" / "embeddings" / "video_emb_small.npy"


def main() -> None:
    df = pd.read_csv(META_CSV)
    caption_map = dict(zip(df["video_id"].astype(str), df["caption"].astype(str)))

    index = faiss.read_index(str(INDEX_PATH))
    with open(IDMAP_PATH, "r") as f:
        ids = json.load(f)["video_ids"]

    X = np.load(EMB_PATH).astype("float32")
    faiss.normalize_L2(X)

    query_row = 0
    topk = 6  # one extra to drop self

    q = X[query_row].reshape(1, -1)
    scores, idx = index.search(q, topk)

    query_id = ids[query_row]
    print(f"\nQuery: {query_id}")
    print(f"Caption: {caption_map.get(query_id, 'N/A')}\n")

    print("Top matches (excluding itself):")
    shown = 0
    for i, s in zip(idx[0], scores[0]):
        i = int(i)
        if i == query_row:
            continue

        vid = ids[i]
        cap = caption_map.get(vid, "N/A")
        shown += 1
        print(f"{shown:02d}. {vid}  score={float(s):.4f}")
        print(f"    caption: {cap}")

        if shown == 5:
            break


if __name__ == "__main__":
    main()
