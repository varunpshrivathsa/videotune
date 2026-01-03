from __future__ import annotations

from pathlib import Path
import subprocess
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]

META_CSV = PROJECT_ROOT / "data" / "processed" / "metadata.csv"
VIDEO_DIR = PROJECT_ROOT / "data" / "video"
AUDIO_DIR = PROJECT_ROOT / "data" / "audio"

SR = 16000

def ffmpeg_extract(in_path: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_path),
        "-vn",
        "-ac", "1",
        "-ar", str(SR),
        "-f", "wav",
        str(out_path),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

def main() -> None:
    df = pd.read_csv(META_CSV)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    missing = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        vid = str(row["video_id"])
        video_path = VIDEO_DIR / f"{vid}.mp4"   # adjust if your extension differs
        audio_path = AUDIO_DIR / f"{vid}.wav"

        if audio_path.exists():
            continue
        if not video_path.exists():
            missing += 1
            continue

        ffmpeg_extract(video_path, audio_path)

    print(f"Done. Missing videos: {missing}")

if __name__ == "__main__":
    main()
