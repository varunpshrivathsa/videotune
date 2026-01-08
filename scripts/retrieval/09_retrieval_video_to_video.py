from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import faiss
import argparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---- Inputs (produced by your earlier scripts) ----
EMB_DIR = PROJECT_ROOT / "data" / "embeddings"
IND_DIR = PROJECT_ROOT / "data" / "faiss"
PROC_DIR = PROJECT_ROOT / "data" / "processed"


VIDEO_NPY = EMB_DIR / "video_clip_train7k.npy"
VIDEO_EMB_IDMAP = EMB_DIR / "video_clip_train7k_idmap.json"

VIDEO_INDEX = IND_DIR / "video_clip_train7k.index"
VIDEO_INDEX_IDMAP = IND_DIR / "video_clip_train7k.index_idmap.json"

META_CSV = PROC_DIR / "video_caption_map.csv"  # has video_id, video_path, caption

def load_idmap(path: Path) -> list[str]:
    with open(path, "r") as f:
        meta = json.load(f)
    if "video_ids" not in meta:
        raise RuntimeError(f"Missing 'video_ids' in {path}")
    return [str(x) for x in meta["video_ids"]]


def build_lookup_from_meta(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    """
    Returns:
      lookup[video_id] = {"caption": ..., "video_path": ...}
    """
    lookup: dict[str, dict[str, str]] = {}
    for _, r in df.iterrows():
        vid = str(r.get("video_id", ""))
        if not vid:
            continue
        lookup[vid] = {
            "caption": str(r.get("caption", "")),
            "video_path": str(r.get("video_path", "")),
        }
    return lookup


def resolve_query_index(
    *,
    query_video_id: str | None,
    query_row: int | None,
    index_video_ids: list[str],
) -> int:
    if query_video_id is not None:
        try:
            return index_video_ids.index(query_video_id)
        except ValueError as e:
            raise RuntimeError(f"video_id '{query_video_id}' not found in index idmap") from e

    if query_row is not None:
        if query_row < 0 or query_row >= len(index_video_ids):
            raise RuntimeError(f"query_row out of range: {query_row} (0..{len(index_video_ids)-1})")
        return query_row

    # default: first row
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Video→Video retrieval using CLIP video embeddings + FAISS.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--video_id", type=str, default=None, help="Query by video_id (e.g., video1234)")
    g.add_argument("--row", type=int, default=None, help="Query by row index into the embedding matrix")
    ap.add_argument("--topk", type=int, default=10, help="Number of neighbors to return (excluding self)")
    args = ap.parse_args()

    # ---- Load FAISS index ----
    if not VIDEO_INDEX.exists():
        raise RuntimeError(f"Missing FAISS index: {VIDEO_INDEX}. Build it with 06_build_faiss_video_index.py")

    index = faiss.read_index(str(VIDEO_INDEX))

    # ---- Load idmap for the index (row -> video_id) ----
    if VIDEO_INDEX_IDMAP.exists():
        index_video_ids = load_idmap(VIDEO_INDEX_IDMAP)
    else:
        # fallback to embedding idmap if needed
        index_video_ids = load_idmap(VIDEO_EMB_IDMAP)

    # ---- Load embeddings (to get the query vector) ----
    if not VIDEO_NPY.exists():
        raise RuntimeError(f"Missing embeddings: {VIDEO_NPY}. Build them with 04_build_video_embeddings_CLIP.py")

    X = np.load(VIDEO_NPY).astype("float32")

    if X.shape[0] != len(index_video_ids):
        raise RuntimeError(
            f"Mismatch: embeddings rows={X.shape[0]} vs idmap ids={len(index_video_ids)} "
            f"(check your .npy and idmap/index files)"
        )

    # Ensure normalized (safe even if already)
    faiss.normalize_L2(X)

    # ---- Metadata lookup (caption/video_path) ----
    meta_lookup: dict[str, dict[str, str]] = {}
    if META_CSV.exists():
        df_meta = pd.read_csv(META_CSV)
        meta_lookup = build_lookup_from_meta(df_meta)

    # ---- Pick query ----
    q_idx = resolve_query_index(
        query_video_id=args.video_id,
        query_row=args.row,
        index_video_ids=index_video_ids,
    )
    q_vid = index_video_ids[q_idx]

    qvec = X[q_idx : q_idx + 1]  # shape (1, D)

    # Ask for one extra because the nearest neighbor is usually itself
    topk = max(1, int(args.topk))
    D, I = index.search(qvec, topk + 1)

    # ---- Print results ----
    print(f"\nQUERY: row={q_idx} video_id={q_vid}")
    if q_vid in meta_lookup:
        cap = meta_lookup[q_vid].get("caption", "")
        vpath = meta_lookup[q_vid].get("video_path", "")
        if cap:
            print("caption:", cap)
        if vpath:
            print("video_path:", vpath)

    print("\nRESULTS (excluding self):")
    rank = 0
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        # drop self match
        if idx == q_idx:
            continue
        rank += 1
        vid = index_video_ids[idx]
        line = f"{rank:02d}. score={float(score):.4f}  video_id={vid}"
        print(line)

        if vid in meta_lookup:
            cap = meta_lookup[vid].get("caption", "")
            vpath = meta_lookup[vid].get("video_path", "")
            if cap:
                print("    caption:", cap)
            if vpath:
                print("    video_path:", vpath)

        if rank >= topk:
            break

    print("")  # newline


if __name__ == "__main__":
    main()