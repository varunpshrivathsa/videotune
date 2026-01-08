from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import faiss

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMB_DIR = PROJECT_ROOT / "data" / "embeddings"
IND_DIR = PROJECT_ROOT / "data" / "faiss"
IND_DIR.mkdir(parents=True, exist_ok=True)


VIDEO_NPY = EMB_DIR / "video_clip_train7k.npy"
VIDEO_IDMAP = EMB_DIR / "video_clip_train7k_idmap.json"

INDEX_PATH = IND_DIR / "video_clip_train7k.index"
INDEX_IDMAP_OUT = IND_DIR / "video_clip_train7k.index_idmap.json"


def main() -> None:
    # Load embeddings
    X = np.load(VIDEO_NPY).astype("float32")

    # Load id map
    with open(VIDEO_IDMAP, "r") as f:
        meta = json.load(f)
    video_ids = meta["video_ids"]

    if X.shape[0] != len(video_ids):
        raise RuntimeError(f"Mismatch: X has {X.shape[0]} rows but idmap has {len(video_ids)} ids")

    # If not normalized, normalize (safe to do even if already normalized)
    faiss.normalize_L2(X)

    d = X.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine similarity when vectors are normalized
    index.add(X)

    faiss.write_index(index, str(INDEX_PATH))

    with open(INDEX_IDMAP_OUT, "w") as f:
        json.dump(
            {
                "video_ids": video_ids,
                "index": "IndexFlatIP",
                "dim": d,
                "normalized": True,
                "source_npy": str(VIDEO_NPY),
            },
            f,
            indent=2,
        )

    print("Saved index:", INDEX_PATH)
    print("Saved idmap:", INDEX_IDMAP_OUT)
    print("Index size:", index.ntotal, "dim:", d)


if __name__ == "__main__":
    main()
