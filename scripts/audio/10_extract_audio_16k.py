from __future__ import annotations

from pathlib import Path
import subprocess
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]

META_CSV = PROJECT_ROOT / "data" / "processed" / "video_caption_map.csv"  # adjust if needed
VIDEO_DIR = PROJECT_ROOT / "data" / "video"
OUT_DIR = PROJECT_ROOT / "data" / "audio_wav"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_one(video_path: Path, wav_path: Path) -> None:
    """
    Extract mono 16k wav. Skip if output exists and looks non-empty.
    """
    if wav_path.exists() and wav_path.stat().st_size > 1024:
        return

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",                 # no video
        "-ac", "1",            # mono
        "-ar", "16000",        # 16 kHz
        "-f", "wav",
        str(wav_path),
    ]
    # quiet but still errors
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def main() -> None:
    df = pd.read_csv(META_CSV)
    # Must have a "video_id" column and optionally "video_path" or similar.
    # If you only have video_id, assume file is VIDEO_DIR / f"{video_id}.mp4"
    video_ids = df["video_id"].astype(str).tolist()

    missing = 0
    for vid in tqdm(video_ids, desc="Extracting audio"):
        # try mp4 by default; change extension if your dataset differs
        vpath = VIDEO_DIR / f"{vid}.mp4"
        if not vpath.exists():
            # fallback: maybe .webm or .avi
            candidates = list(VIDEO_DIR.glob(f"{vid}.*"))
            if not candidates:
                missing += 1
                continue
            vpath = candidates[0]

        wav_path = OUT_DIR / f"{vid}.wav"
        extract_one(vpath, wav_path)

    print(f"Done. wav_dir={OUT_DIR}")
    print(f"Missing videos: {missing}")


if __name__ == "__main__":
    main()
