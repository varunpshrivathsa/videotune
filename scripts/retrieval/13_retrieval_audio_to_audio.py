from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import faiss

PROJECT_ROOT = Path(__file__).resolve().parents[2]

META_CSV = PROJECT_ROOT / "data" / "processed" / "video_caption_map.csv"

EMB_PATH = PROJECT_ROOT / "data" / "embeddings" / "audio_wav2vec_train7k.npy"
IDMAP_PATH = PROJECT_ROOT / "data" / "faiss" / "audio_wav2vec_train7k.index_idmap.json"
INDEX_PATH = PROJECT_ROOT / "data" / "faiss" / "audio_wav2vec_train7k.index"


def main(video_id: str, topk: int = 10) -> None:
    df = pd.read_csv(META_CSV)
    caption_map = dict(zip(df["video_id"].astype(str), df["caption"].astype(str)))

    X = np.load(EMB_PATH).astype("float32")
    faiss.normalize_L2(X)

    index = faiss.read_index(str(INDEX_PATH))
    with open(IDMAP_PATH, "r") as f:
        ids = json.load(f)["video_ids"]

    # find query row
    try:
        q_idx = ids.index(video_id)
    except ValueError:
        raise SystemExit(f"video_id not found in idmap: {video_id}")

    q = X[q_idx].reshape(1, -1)
    scores, idxs = index.search(q, topk + 1)  # +1 to drop self
    scores = scores[0]
    idxs = idxs[0]

    print(f"\nQUERY video_id={video_id}")
    print("caption:", caption_map.get(video_id, "<no caption>"))

    rank = 0
    for s, j in zip(scores, idxs):
        if j < 0:
            continue
        vid = ids[j]
        if vid == video_id:
            continue
        rank += 1
        print(f"{rank:02d}. score={float(s):.4f}  video_id={vid}")
        print(f"    caption: {caption_map.get(vid, '<no caption>')}")
        if rank >= topk:
            break


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_id", type=str, required=True)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()
    main(args.video_id, args.topk)
