from pathlib import Path
import pandas as pd

cap_df = pd.read_csv("../../data/caption/caption_train7k.csv")
video_dir = Path("../../data/video")

video_ids_on_disk = {p.stem for p in video_dir.glob("*.mp4")}

cap_df["has_video"] = cap_df["video_id"].isin(video_ids_on_disk)

print("Total captions:", len(cap_df))
print("Videos on disk:", len(video_ids_on_disk))
print("Matched:", cap_df["has_video"].sum())
print("Missing videos:", (~cap_df["has_video"]).sum())

cap_df[~cap_df["has_video"]].head()
