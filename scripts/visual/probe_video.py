from __future__ import annotations

from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_PATH = PROJECT_ROOT / "data" / "video" / "video0.mp4"


def probe(video_path: Path) -> None:
    """
    Opens a video and prints basic metadata from OpenCV.

    Why this matters:
    - FPS tells us how many frames per second exist in the file.
    - Frame count *may* be inaccurate for some codecs, but it's a useful estimate.
    - Duration = frame_count / fps (approx)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    duration = (frame_count / fps) if fps and fps > 0 else None

    print(f"Video: {video_path}")
    print(f"FPS: {fps}")
    print(f"Frame count (reported): {frame_count}")
    print(f"Resolution: {int(width)}x{int(height)}")
    print(f"Approx duration (sec): {duration}")

    cap.release()


if __name__ == "__main__":
    probe(VIDEO_PATH)
