from __future__ import annotations

from pathlib import Path
import cv2
import torch
import clip
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_PATH = PROJECT_ROOT / "data" / "video" / "video0.mp4"

def main():
    video_path = VIDEO_PATH

    # 1) Read the very first frame
    cap = cv2.VideoCapture(str(video_path))
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Could not read first frame")

    # 2) Convert BGR -> RGB (OpenCV uses BGR)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # 3) Convert numpy -> PIL (CLIP preprocess expects PIL)
    img = Image.fromarray(frame_rgb)

    # 4) Load CLIP model + preprocess
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    model.eval()

    # 5) Preprocess + encode
    x = preprocess(img).unsqueeze(0).to(DEVICE)  # shape: (1, 3, 224, 224)
    with torch.no_grad():
        feat = model.encode_image(x).float()      # shape: (1, D)

    # 6) Normalize (unit length) for cosine similarity
    feat = feat / feat.norm(dim=-1, keepdim=True)

    print("Device:", DEVICE)
    print("Embedding shape:", tuple(feat.shape))
    print("Embedding norm:", feat.norm(dim=-1).item())

if __name__ == "__main__":
    main()
