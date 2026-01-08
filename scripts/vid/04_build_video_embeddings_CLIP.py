from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MAP_CSV = PROJECT_ROOT / "data" / "processed" / "video_caption_map.csv"
OUT_DIR = PROJECT_ROOT / "data" / "embeddings"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMB_NPY = OUT_DIR / "video_clip_train7k.npy"
IDMAP_JSON = OUT_DIR / "video_clip_train7k_idmap.json"


MODEL_NAME = "openai/clip-vit-base-patch32"
NUM_FRAMES = 16
BATCH_FRAMES = 64  # frames per batch for CLIP (adjust if OOM)


def sample_uniform_frames(video_path: Path, num_frames: int) -> list[Image.Image]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"No frames in video: {video_path}")

    # pick indices uniformly over [0, frame_count-1]
    idxs = np.linspace(0, frame_count - 1, num_frames).round().astype(int)

    images: list[Image.Image] = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            # fallback: skip; will handle later
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        images.append(Image.fromarray(frame_rgb))

    cap.release()

    if len(images) == 0:
        raise RuntimeError(f"Failed to sample any frames: {video_path}")

    # If we got fewer than num_frames (rare), pad by repeating last frame
    while len(images) < num_frames:
        images.append(images[-1].copy())

    return images[:num_frames]


@torch.inference_mode()
def main() -> None:
    df = pd.read_csv(MAP_CSV)
    # Your CSV currently has ../.. paths; normalize to absolute paths from project root
    # If your video_path is already "data/video/xxx.mp4", this also works.
    df["video_path_abs"] = df["video_path"].apply(
        lambda p: PROJECT_ROOT / p
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    video_ids: list[str] = []
    video_embs: list[np.ndarray] = []

    frame_buffer: list[Image.Image] = []
    owner_buffer: list[int] = []  # which video index each frame belongs to

    # We'll accumulate frame embeddings then mean-pool per video.
    # Approach:
    # - sample frames for a video
    # - push to frame buffer
    # - run CLIP in batches over frames
    # - collect per-frame embeddings, pool per video

    per_video_frame_embs: list[list[np.ndarray]] = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        vid = str(row["video_id"])
        vpath = Path(row["video_path_abs"])

        frames = sample_uniform_frames(vpath, NUM_FRAMES)

        # store placeholder list for this video
        per_video_frame_embs.append([])

        for im in frames:
            frame_buffer.append(im)
            owner_buffer.append(i)

        # flush if enough frames
        if len(frame_buffer) >= BATCH_FRAMES:
            inputs = processor(images=frame_buffer, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            feats = model.get_image_features(**inputs)  # (B, D)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            feats = feats.detach().cpu().numpy()

            for f, owner in zip(feats, owner_buffer):
                per_video_frame_embs[owner].append(f)

            frame_buffer.clear()
            owner_buffer.clear()

    # flush remaining frames
    if len(frame_buffer) > 0:
        inputs = processor(images=frame_buffer, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        feats = model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        feats = feats.detach().cpu().numpy()

        for f, owner in zip(feats, owner_buffer):
            per_video_frame_embs[owner].append(f)

        frame_buffer.clear()
        owner_buffer.clear()

    # mean-pool frames -> one vector per video
    for i, row in df.iterrows():
        vid = str(row["video_id"])
        frames = per_video_frame_embs[i]
        if len(frames) == 0:
            raise RuntimeError(f"No frame embeddings collected for {vid}")
        v = np.max(np.stack(frames, axis=0), axis=0).astype("float32")
        # normalize again
        v = v / np.linalg.norm(v)
        video_ids.append(vid)
        video_embs.append(v)

    X = np.stack(video_embs, axis=0).astype("float32")
    np.save(EMB_NPY, X)

    with open(IDMAP_JSON, "w") as f:
        json.dump({"video_ids": video_ids, "model": MODEL_NAME, "num_frames": NUM_FRAMES}, f, indent=2)

    print("Saved embeddings:", EMB_NPY, "shape:", X.shape)
    print("Saved idmap:", IDMAP_JSON)


if __name__ == "__main__":
    main()
