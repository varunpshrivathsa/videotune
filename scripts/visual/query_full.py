from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import faiss

PROJECT_ROOT = Path(__file__).resolve().parents[1]

META_CSV = PROJECT_ROOT / "data" / "processed" / "metadata.csv"
INDEX_PATH = PROJECT_ROOT / "data" / "index" / "video_full.index"
IDMAP_PATH = PROJECT_ROOT / "data" / "index" / "video_full_idmap.json"
EMB_PATH = PROJECT_ROOT / "data" / "embeddings" / "full" / "video_emb.npy"


def main() -> None:
    df = pd.read_csv(META_CSV)
    caption_map = dict(zip(df["video_id"].astype(str), df["caption"].astype(str)))

    index = faiss.read_index(str(INDEX_PATH))
    ids = json.loads(Path(IDMAP_PATH).read_text())["video_ids"]

    X = np.load(EMB_PATH).astype("float32")
    faiss.normalize_L2(X)

    # Query by row index (0 == video0)
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
        shown += 1
        print(f"{shown:02d}. {vid}  score={float(s):.4f}")
        print(f"    caption: {caption_map.get(vid, 'N/A')}")
        if shown == 5:
            break


if __name__ == "__main__":
    main()
