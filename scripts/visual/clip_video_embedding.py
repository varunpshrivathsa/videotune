from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import torch
import clip
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIDEO_PATH = PROJECT_ROOT / "data" / "video" / "video0.mp4"

def sample_frames(video_path: Path, target_fps: float = 1.0, max_frames: int = 32) -> list[np.ndarray]:
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
def embed_video(video_path: Path, target_fps: float = 1.0, max_frames: int = 32) -> np.ndarray:
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    model.eval()

    frames_bgr = sample_frames(video_path, target_fps=target_fps, max_frames=max_frames)
    if len(frames_bgr) == 0:
        raise RuntimeError(f"No frames sampled from {video_path}")

    # Convert frames to a batch tensor
    imgs = []
    for frame_bgr in frames_bgr:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgs.append(preprocess(img))

    batch = torch.stack(imgs).to(DEVICE)  # (T, 3, 224, 224)

    # Encode each frame -> (T, 512)
    feats = model.encode_image(batch).float()

    # Normalize each frame embedding
    feats = feats / feats.norm(dim=-1, keepdim=True)

    # Average across time -> (512,)
    vid = feats.mean(dim=0)

    # Normalize final video embedding
    vid = vid / vid.norm()

    return vid.cpu().numpy().astype("float32")

def main():
    vp = VIDEO_PATH
    vec = embed_video(vp, target_fps=1.0, max_frames=32)
    print("Device:", DEVICE)
    print("Video embedding shape:", vec.shape)
    print("Video embedding norm:", float(np.linalg.norm(vec)))

if __name__ == "__main__":
    main()
