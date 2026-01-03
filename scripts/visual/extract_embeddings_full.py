from __future__ import annotations

from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
import torch
import clip
from PIL import Image


# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
META_CSV = PROJECT_ROOT / "data" / "processed" / "metadata.csv"

OUT_ROOT = PROJECT_ROOT / "data" / "embeddings" / "full"
CHUNK_DIR = OUT_ROOT / "chunks"
MANIFEST = OUT_ROOT / "manifest.json"

FINAL_EMB = OUT_ROOT / "video_emb.npy"
FINAL_IDS = OUT_ROOT / "video_ids.json"


# ---------- Config ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B/32"
TARGET_FPS = 1.0
MAX_FRAMES = 32

CHUNK_SIZE = 256  # embeddings saved every 256 videos


def sample_frames(video_path: Path, target_fps: float, max_frames: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps is None or native_fps <= 0:
        native_fps = 30.0

    stride = max(int(round(native_fps / target_fps)), 1)

    frames: list[np.ndarray] = []
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
    frames_bgr = sample_frames(video_path, TARGET_FPS, MAX_FRAMES)
    if len(frames_bgr) == 0:
        raise RuntimeError(f"No frames sampled from {video_path}")

    imgs = []
    for frame_bgr in frames_bgr:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgs.append(preprocess(img))

    batch = torch.stack(imgs).to(DEVICE)              # (T, 3, 224, 224)
    feats = model.encode_image(batch).float()         # (T, 512)
    feats = feats / feats.norm(dim=-1, keepdim=True)  # normalize each frame

    vid = feats.mean(dim=0)                           # (512,)
    vid = vid / vid.norm()                            # normalize final video vector
    return vid.cpu().numpy().astype("float32")


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {"completed_chunks": []}


def save_manifest(m: dict) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=2))


def chunk_paths(start_idx: int) -> tuple[Path, Path]:
    emb_path = CHUNK_DIR / f"emb_{start_idx:05d}.npy"
    ids_path = CHUNK_DIR / f"ids_{start_idx:05d}.json"
    return emb_path, ids_path


def main() -> None:
    if not META_CSV.exists():
        raise FileNotFoundError(f"Missing {META_CSV}. Run build_metadata_msrvtt.py first.")

    df = pd.read_csv(META_CSV)
    n = len(df)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    completed = set(manifest.get("completed_chunks", []))

    model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
    model.eval()

    # Process in chunks: [0..255], [256..511], ...
    for start in range(0, n, CHUNK_SIZE):
        if start in completed:
            print(f"Skipping chunk starting {start} (already done)")
            continue

        end = min(start + CHUNK_SIZE, n)
        chunk_df = df.iloc[start:end]

        embs: list[np.ndarray] = []
        ids: list[str] = []
        failures: list[dict] = []

        print(f"\n--- Chunk {start}..{end-1} ---")

        for row_idx, row in chunk_df.iterrows():
            vid = str(row["video_id"])
            vpath = Path(row["video_path"])

            try:
                if not vpath.exists():
                    raise FileNotFoundError(f"Missing file: {vpath}")

                vec = embed_video(model, preprocess, vpath)
                embs.append(vec)
                ids.append(vid)

                if len(ids) % 25 == 0:
                    print(f"  progress: {len(ids)}/{len(chunk_df)} (latest={vid})")

            except Exception as e:
                failures.append({"video_id": vid, "video_path": str(vpath), "error": repr(e)})

        # Save chunk
        emb_path, ids_path = chunk_paths(start)
        np.save(emb_path, np.vstack(embs).astype("float32"))
        ids_path.write_text(json.dumps(ids, indent=2))

        # Save failures (if any)
        if failures:
            (CHUNK_DIR / f"fail_{start:05d}.json").write_text(json.dumps(failures, indent=2))

        # Mark chunk done
        manifest["completed_chunks"] = sorted(set(manifest.get("completed_chunks", []) + [start]))
        save_manifest(manifest)

        print(f"Saved chunk embeddings: {emb_path}  rows={len(ids)}")
        if failures:
            print(f"Failures in this chunk: {len(failures)} (see fail_{start:05d}.json)")

    # Merge chunks into final arrays
    print("\nMerging chunks into final files...")
    all_embs: list[np.ndarray] = []
    all_ids: list[str] = []

    for start in range(0, n, CHUNK_SIZE):
        emb_path, ids_path = chunk_paths(start)
        if not emb_path.exists() or not ids_path.exists():
            raise RuntimeError(f"Missing chunk files for start={start}. Run again to complete.")

        Xc = np.load(emb_path)
        idc = json.loads(ids_path.read_text())

        if Xc.shape[0] != len(idc):
            raise RuntimeError(f"Chunk mismatch at start={start}: {Xc.shape[0]} vs {len(idc)}")

        all_embs.append(Xc)
        all_ids.extend(idc)

    X = np.vstack(all_embs).astype("float32")

    np.save(FINAL_EMB, X)
    FINAL_IDS.write_text(json.dumps(all_ids, indent=2))

    print(f"Final embeddings saved: {FINAL_EMB} shape={X.shape} dtype={X.dtype}")
    print(f"Final ids saved       : {FINAL_IDS} len={len(all_ids)}")


if __name__ == "__main__":
    main()
