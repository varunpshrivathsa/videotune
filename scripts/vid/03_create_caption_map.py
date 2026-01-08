from pathlib import Path
import pandas as pd

cap_df = pd.read_csv("../../data/caption/caption_train7k.csv")

cap_df["video_path"] = cap_df["video_id"].apply(lambda vid: f"../../data/video/{vid}.mp4")
cap_df = cap_df[["video_id", "video_path", "caption"]]

out = Path("../../data/processed/video_caption_map.csv")
out.parent.mkdir(parents=True, exist_ok=True)
cap_df.to_csv(out, index=False)

print("Saved:", out)
print(cap_df.head())
