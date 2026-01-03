from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import faiss

PROJECT_ROOT = Path(__file__).resolve().parents[2]

META_CSV = PROJECT_ROOT / "data" / "processed" / "metadata.csv"
EMB_PATH = PROJECT_ROOT / "data" / "embeddings" / "audio_emb_full.npy"

OUT_EMB = PROJECT_ROOT / "data" / "embeddings" / "audio_emb_full_valid.npy"
INDEX_PATH = PROJECT_ROOT / "data" / "index" / "audio_full_valid.index"
IDMAP_PATH = PROJECT_ROOT / "data" / "index" / "audio_full_valid_idmap.json"

EPS = 1e-8


def main() -> None:
    df = pd.read_csv(META_CSV)
    ids = df["video_id"].astype(str).tolist()

    X = np.load(EMB_PATH).astype("float32")
    norms = np.linalg.norm(X, axis=1)
    keep = norms >= EPS

    Xv = X[keep]
    idsv = [vid for vid, k in zip(ids, keep) if k]

    faiss.normalize_L2(Xv)

    index = faiss.IndexFlatIP(Xv.shape[1])
    index.add(Xv)

    OUT_EMB.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_EMB, Xv)

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))

    with open(IDMAP_PATH, "w") as f:
        json.dump({"video_ids": idsv}, f)

    print(f"Kept valid audio: {len(idsv)} / {len(ids)}")
    print(f"Saved emb: {OUT_EMB}  shape={Xv.shape}")
    print(f"Saved index: {INDEX_PATH}")
    print(f"Saved idmap: {IDMAP_PATH}")


if __name__ == "__main__":
    main()
