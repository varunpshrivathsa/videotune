from pathlib import Path
import pandas as pd

# Load raw captions
cap_df = pd.read_csv("../../data/caption/caption_train7k.csv")

# Keep only needed columns
cap_df = cap_df[["video_id", "caption"]]

# IMPORTANT: MSR-VTT captions are concatenated — keep only ONE
cap_df["caption"] = (
    cap_df["caption"]
    .astype(str)
    .str.split(r"\.\s+")
    .str[0]
    .str.strip()
)

# Build video path
cap_df["video_path"] = cap_df["video_id"].apply(
    lambda vid: f"data/video/{vid}.mp4"
)

# Reorder
cap_df = cap_df[["video_id", "video_path", "caption"]]

# Save
out = Path("../../data/processed/video_caption_map.csv")
out.parent.mkdir(parents=True, exist_ok=True)
cap_df.to_csv(out, index=False)

print("Saved:", out)
print("Rows:", len(cap_df))
print(cap_df.head())

