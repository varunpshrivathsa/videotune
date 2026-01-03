from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import faiss

PROJECT_ROOT = Path(__file__).resolve().parents[2]

META_CSV = PROJECT_ROOT / "data" / "processed" / "metadata.csv"
INDEX_PATH = PROJECT_ROOT / "data" / "index" / "audio_full.index"
IDMAP_PATH = PROJECT_ROOT / "data" / "index" / "audio_full_idmap.json"
EMB_PATH = PROJECT_ROOT / "data" / "embeddings" / "audio_emb_full.npy"

def main() -> None:
    df = pd.read_csv(META_CSV)
    caption_map = dict(zip(df["video_id"].astype(str), df["caption"].astype(str)))

    index = faiss.read_index(str(INDEX_PATH))
    with open(IDMAP_PATH, "r") as f:
        ids = json.load(f)["video_ids"]

    X = np.load(EMB_PATH).astype("float32")
    faiss.normalize_L2(X)

    query_row = 0
    topk = 10
    q = X[query_row].reshape(1, -1)

    D, I = index.search(q, topk)

    qid = ids[query_row]
    print(f"\nQUERY video_id={qid}")
    print(f"caption: {caption_map.get(str(qid), '')}\n")

    shown = 0
    for idx, score in zip(I[0], D[0]):
        vid = ids[idx]
        if vid == qid:
            continue
        shown += 1
        print(f"{shown:02d}. score={score:.4f}  video_id={vid}")
        print(f"    caption: {caption_map.get(str(vid), '')}")
        if shown >= topk - 1:
            break

if __name__ == "__main__":
    main()
