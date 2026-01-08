from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import faiss

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMB_DIR = PROJECT_ROOT / "data" / "embeddings"
FAISS_DIR = PROJECT_ROOT / "data" / "faiss"
FAISS_DIR.mkdir(parents=True, exist_ok=True)

EMB_PATH = EMB_DIR / "audio_wav2vec_train7k.npy"
IDMAP_IN = EMB_DIR / "audio_wav2vec_train7k_idmap.json"

INDEX_PATH = FAISS_DIR / "audio_wav2vec_train7k.index"
IDMAP_OUT = FAISS_DIR / "audio_wav2vec_train7k.index_idmap.json"


def main() -> None:
    X = np.load(EMB_PATH).astype("float32")
    # Ensure L2-normalized (cosine via inner product)
    faiss.normalize_L2(X)

    d = X.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(X)

    faiss.write_index(index, str(INDEX_PATH))

    with open(IDMAP_IN, "r") as f:
        idmap = json.load(f)
    with open(IDMAP_OUT, "w") as f:
        json.dump(idmap, f, indent=2)

    print(f"Saved index: {INDEX_PATH}")
    print(f"Saved idmap: {IDMAP_OUT}")
    print(f"ntotal={index.ntotal}, dim={d}")


if __name__ == "__main__":
    main()
