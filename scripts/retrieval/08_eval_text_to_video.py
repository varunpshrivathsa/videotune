from __future__ import annotations

from pathlib import Path
import argparse
import json
import numpy as np
import faiss


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMB_DIR = PROJECT_ROOT / "data" / "embeddings"
IND_DIR = PROJECT_ROOT / "data" / "faiss"

TEXT_NPY = EMB_DIR / "text_clip_train7k.npy"
TEXT_IDMAP = EMB_DIR / "text_clip_train7k_idmap.json"

INDEX_PATH = IND_DIR / "video_clip_train7k.index"
INDEX_IDMAP = IND_DIR / "video_clip_train7k.index_idmap.json"


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    # rank is 1-based; hit if rank <= k
    return float(np.mean(ranks <= k))


def mrr_at_k(ranks: np.ndarray, k: int) -> float:
    # reciprocal rank if found within k, else 0
    rr = np.where(ranks <= k, 1.0 / ranks, 0.0)
    return float(np.mean(rr))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=50, help="Evaluate up to this top-k (used for recall + capped rank)")
    ap.add_argument("--max_queries", type=int, default=0, help="If >0, evaluate only first N queries (debug/speed)")
    args = ap.parse_args()

    topk = int(args.topk)
    if topk <= 0:
        raise ValueError("--topk must be > 0")

    # ---- Load text embeddings + text idmap (ground truth order)
    T = np.load(TEXT_NPY).astype("float32")
    with open(TEXT_IDMAP, "r") as f:
        text_meta = json.load(f)
    gt_video_ids = text_meta["video_ids"]

    if T.shape[0] != len(gt_video_ids):
        raise RuntimeError(f"Mismatch: text embeddings rows={T.shape[0]} vs text idmap ids={len(gt_video_ids)}")

    # Optional subset
    n = T.shape[0] if args.max_queries <= 0 else min(args.max_queries, T.shape[0])
    T = T[:n]
    gt_video_ids = gt_video_ids[:n]

    # ---- Load FAISS video index + its idmap (FAISS index order)
    index = faiss.read_index(str(INDEX_PATH))
    with open(INDEX_IDMAP, "r") as f:
        ind_meta = json.load(f)
    index_video_ids = ind_meta["video_ids"]

    if index.ntotal != len(index_video_ids):
        raise RuntimeError(f"Mismatch: index.ntotal={index.ntotal} vs index idmap ids={len(index_video_ids)}")

    # ---- Build fast lookup: video_id -> position in index list
    # (We need this to check whether retrieved ids match ground truth)
    # Note: ground-truth video_id should exist in index_video_ids.
    index_pos = {vid: i for i, vid in enumerate(index_video_ids)}

    missing = [vid for vid in gt_video_ids if vid not in index_pos]
    if missing:
        raise RuntimeError(f"{len(missing)} ground-truth video_ids not found in index idmap (example: {missing[0]})")

    # ---- Search: text embedding queries against video index
    # Ensure normalized (safe even if already normalized)
    faiss.normalize_L2(T)

    # scores: (n, topk)  idxs: (n, topk)
    scores, idxs = index.search(T, topk)

    # ---- Compute rank of correct video within topk (1-based); if not found -> topk+1
    ranks = np.full((n,), fill_value=topk + 1, dtype=np.int32)

    for i in range(n):
        gt_vid = gt_video_ids[i]
        gt_pos = index_pos[gt_vid]

        hits = np.where(idxs[i] == gt_pos)[0]
        if hits.size > 0:
            ranks[i] = int(hits[0]) + 1  # 1-based rank

    # ---- Metrics
    r1 = recall_at_k(ranks, 1)
    r5 = recall_at_k(ranks, 5)
    r10 = recall_at_k(ranks, 10)
    r50 = recall_at_k(ranks, min(50, topk))
    mrr10 = mrr_at_k(ranks, min(10, topk))
    med_rank = float(np.median(ranks))  # capped median rank

    print("\n=== Text -> Video Retrieval Sanity Eval ===")
    print(f"Queries evaluated: {n}")
    print(f"TopK searched: {topk}")
    print("")
    print(f"Recall@1 : {r1:.4f}")
    print(f"Recall@5 : {r5:.4f}")
    print(f"Recall@10: {r10:.4f}")
    if topk >= 50:
        print(f"Recall@50: {r50:.4f}")
    print(f"MRR@10   : {mrr10:.4f}")
    print(f"Median Rank (capped at {topk+1}): {med_rank:.1f}")

    # Optional: show a few failures (where rank is topk+1)
    fail_idx = np.where(ranks == topk + 1)[0]
    if fail_idx.size > 0:
        print(f"\nMisses within top-{topk}: {fail_idx.size} ({fail_idx.size/n:.2%})")
        show = fail_idx[:5]
        for i in show:
            print(f"- example miss: query_row={i}, gt_video_id={gt_video_ids[i]}")
    else:
        print(f"\nNo misses within top-{topk} 🎯")


if __name__ == "__main__":
    main()
