from __future__ import annotations

from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_PATH = PROJECT_ROOT / "data" / "video" / "video0.mp4"

def sample_frames(
    video_path: Path,
    target_fps: float = 1.0,
    max_frames: int = 32,
) -> list:
    """
    Returns a list of sampled frames (as numpy arrays in BGR format).

    Sampling rule:
    - Let native_fps be the video's FPS (e.g., 25).
    - To sample target_fps (e.g., 1 fps), take every `stride` frames where:
        stride = round(native_fps / target_fps)
      and stride is at least 1.

    We also cap the number of returned frames to max_frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps is None or native_fps <= 0:
        native_fps = 30.0  # fallback assumption

    stride = max(int(round(native_fps / target_fps)), 1)

    frames = []
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if idx % stride == 0:
            frames.append(frame)

            if len(frames) >= max_frames:
                break

        idx += 1

    cap.release()
    return frames


def main() -> None:
    vp = VIDEO_PATH
    frames = sample_frames(vp, target_fps=1.0, max_frames=32)
    print(f"Sampled frames: {len(frames)}")
    if frames:
        h, w = frames[0].shape[:2]
        print(f"Frame shape (H,W): {h},{w}")


if __name__ == "__main__":
    main()
