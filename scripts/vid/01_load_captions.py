from datasets import load_dataset
import pandas as pd
from pathlib import Path
import numpy as np

ds = load_dataset("friedrichor/MSR-VTT", "train_7k")
df = ds["train"].to_pandas()

df = df[["video_id", "caption"]].copy()

def flatten_and_join(caps):
    flat = []
    for c in caps:
        # c can be list, tuple, numpy array, or a single string
        if isinstance(c, (list, tuple, np.ndarray)):
            flat.extend([str(x) for x in c])
        else:
            flat.append(str(c))
    return ". ".join(flat)

df = (
    df.groupby("video_id")["caption"]
      .apply(flatten_and_join)
      .reset_index()
)

df["video_id"] = df["video_id"].astype(str)
df["caption"] = df["caption"].str.lower().str.strip()

OUTPUT = Path("../../data/caption/caption_train7k.csv")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print(df.head())
print("Saved:", OUTPUT)
