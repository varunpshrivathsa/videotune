from __future__ import annotations

from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
import torch
import clip
from PIL import Image


# ---------- Paths (robust regardless of working directory) ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
META_CSV = PROJECT_ROOT / "data" / "processed" / "metadata.csv"
OUT_DIR = PROJECT_ROOT / "data" / "embeddings"
OUT_EMB = OUT_DIR / "video_emb_small.npy"
OUT_IDS = OUT_DIR / "video_ids_small.json"


# ---------- Runtime config ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B/32"
N_VIDEOS = 50
TARGET_FPS = 1.0
MAX_FRAMES = 32


def sample_frames(video_path: Path, target_fps: float, max_frames: int) -> list[np.ndarray]:
    """Uniformly sample frames using stride computed from native FPS."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps is None or native_fps <= 0:
        native_fps = 30.0

    stride = max(int(round(native_fps / target_fps)), 1)

    frames = []
    idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if idx % stride == 0:
            frames.append(frame_bgr)
            if len(frames) >= max_frames:
                break

        idx += 1

    cap.release()
    return frames


@torch.no_grad()
def embed_video(model, preprocess, video_path: Path) -> np.ndarray:
    """Video -> (512,) embedding via frame CLIP + mean pooling."""
    frames_bgr = sample_frames(video_path, TARGET_FPS, MAX_FRAMES)
    if len(frames_bgr) == 0:
        raise RuntimeError(f"No frames sampled from {video_path}")

    # Convert frames -> preprocessed tensors
    imgs = []
    for frame_bgr in frames_bgr:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgs.append(preprocess(img))

    batch = torch.stack(imgs).to(DEVICE)   # (T, 3, 224, 224)

    feats = model.encode_image(batch).float()          # (T, 512)
    feats = feats / feats.norm(dim=-1, keepdim=True)   # normalize each frame

    vid = feats.mean(dim=0)                            # (512,)
    vid = vid / vid.norm()                             # normalize final video vector

    return vid.cpu().numpy().astype("float32")


def main() -> None:
    if not META_CSV.exists():
        raise FileNotFoundError(f"Missing metadata: {META_CSV} (run build_metadata_msrvtt.py)")

    df = pd.read_csv(META_CSV).head(N_VIDEOS)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load CLIP once (important: don’t reload for every video)
    model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
    model.eval()

    video_ids: list[str] = []
    embs: list[np.ndarray] = []

    for i, row in df.iterrows():
        vid = str(row["video_id"])
        vpath = Path(row["video_path"])

        if not vpath.exists():
            raise FileNotFoundError(f"Video missing on disk: {vpath}")

        vec = embed_video(model, preprocess, vpath)

        video_ids.append(vid)
        embs.append(vec)

        print(f"[{len(video_ids):02d}/{N_VIDEOS}] embedded {vid}  norm={float(np.linalg.norm(vec)):.4f}")

    X = np.vstack(embs).astype("float32")   # (N, 512)

    np.save(OUT_EMB, X)
    with open(OUT_IDS, "w") as f:
        json.dump(video_ids, f)

    print("\nSaved:")
    print(" -", OUT_EMB, "shape=", X.shape, "dtype=", X.dtype)
    print(" -", OUT_IDS, "len=", len(video_ids))


if __name__ == "__main__":
    main()
