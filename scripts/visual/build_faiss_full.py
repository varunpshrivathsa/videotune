from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import faiss

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMB_PATH = PROJECT_ROOT / "data" / "embeddings" / "full" / "video_emb.npy"
IDS_PATH = PROJECT_ROOT / "data" / "embeddings" / "full" / "video_ids.json"

OUT_DIR = PROJECT_ROOT / "data" / "index"
OUT_INDEX = OUT_DIR / "video_full.index"
OUT_IDMAP = OUT_DIR / "video_full_idmap.json"


def main() -> None:
    if not EMB_PATH.exists() or not IDS_PATH.exists():
        raise FileNotFoundError("Missing full embeddings. Run extract_embeddings_full.py first.")

    X = np.load(EMB_PATH).astype("float32")
    ids = json.loads(IDS_PATH.read_text())

    if X.shape[0] != len(ids):
        raise RuntimeError(f"Mismatch: embeddings rows={X.shape[0]} but ids={len(ids)}")

    faiss.normalize_L2(X)

    d = X.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(X)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(OUT_INDEX))
    OUT_IDMAP.write_text(json.dumps({"video_ids": ids}, indent=2))

    print(f"Built index: {OUT_INDEX}")
    print(f"   ntotal={index.ntotal}, dim={d}")
    print(f"Saved id map: {OUT_IDMAP}")


if __name__ == "__main__":
    main()
