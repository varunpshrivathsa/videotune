from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import faiss

PROJECT_ROOT = Path(__file__).resolve().parents[2]

META_CSV = PROJECT_ROOT / "data" / "processed" / "metadata.csv"
EMB_PATH = PROJECT_ROOT / "data" / "embeddings" / "audio_emb_full.npy"
INDEX_PATH = PROJECT_ROOT / "data" / "index" / "audio_full.index"
IDMAP_PATH = PROJECT_ROOT / "data" / "index" / "audio_full_idmap.json"

def main() -> None:
    df = pd.read_csv(META_CSV)
    video_ids = df["video_id"].astype(str).tolist()

    X = np.load(EMB_PATH).astype("float32")
    faiss.normalize_L2(X)

    d = X.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine similarity via normalized IP
    index.add(X)

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))

    with open(IDMAP_PATH, "w") as f:
        json.dump({"video_ids": video_ids}, f)

    print(f"Saved index: {INDEX_PATH}")
    print(f"Saved idmap: {IDMAP_PATH}")
    print(f"X shape: {X.shape}")

if __name__ == "__main__":
    main()
