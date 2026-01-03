from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
from datasets import load_dataset

# ---- Project paths (single source of truth) ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIDEO_DIR = PROJECT_ROOT / "data" / "video"
OUT_CSV = PROJECT_ROOT / "data" / "processed" / "metadata.csv"

HF_DATASET = "friedrichor/MSR-VTT"
HF_CONFIG = "train_7k"   # later we can also do "test_1k" or "train_9k"


def pick_caption(captions) -> str:
    """
    HF gives captions as a list (often length 20).
    For Phase 1, we pick one caption to keep one row per video.
    """
    if isinstance(captions, list) and len(captions) > 0:
        return str(captions[0])
    # In case the dataset config ever returns a single string
    if isinstance(captions, str):
        return captions
    return ""


def main() -> None:
    # 1) Load Hugging Face split
    ds = load_dataset(HF_DATASET, HF_CONFIG)
    split = ds["train"]

    rows = []
    missing_files = 0

    # 2) Iterate rows and map to local mp4s
    for ex in split:
        video_id = str(ex["video_id"])          # e.g., "video0"
        filename = str(ex["video"])             # e.g., "video0.mp4"
        caption = pick_caption(ex["caption"])   # choose 1 caption

        video_path = VIDEO_DIR / filename

        if not video_path.exists():
            missing_files += 1
            continue

        rows.append(
            {
                "video_id": video_id,
                "video_path": str(video_path.as_posix()),
                "caption": caption,
            }
        )

    # 3) Save to CSV (create folder if needed)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)

    # Safety checks
    if df.empty:
        raise RuntimeError(
            "metadata.csv ended up empty. "
            "Most likely VIDEO_DIR path is wrong or videos are missing."
        )

    # 4) Ensure uniqueness: one row per video_id
    # If duplicates ever happen, keep the first.
    df = df.drop_duplicates(subset=["video_id"], keep="first").reset_index(drop=True)

    df.to_csv(OUT_CSV, index=False)

    print(f"Wrote: {OUT_CSV}  rows={len(df)}")
    print(f"Missing local video files skipped: {missing_files}")


if __name__ == "__main__":
    main()
