from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import Wav2Vec2Processor, Wav2Vec2Model


# =========================
# Paths / Config
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Must contain columns: video_id, caption (caption not required for embedding, just for alignment)
META_CSV = PROJECT_ROOT / "data" / "processed" / "video_caption_map.csv"

# Extracted audio wavs (16kHz mono wavs)
WAV_DIR = PROJECT_ROOT / "data" / "audio_wav"

EMB_DIR = PROJECT_ROOT / "data" / "embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)

OUT_NPY = EMB_DIR / "audio_wav2vec_train7k.npy"
OUT_IDMAP = EMB_DIR / "audio_wav2vec_train7k_idmap.json"
BAD_LOG = EMB_DIR / "audio_wav2vec_bad_wavs.txt"

MODEL_NAME = "facebook/wav2vec2-base-960h"
SAMPLE_RATE = 16000

# Make Wav2Vec2 safe:
MIN_SAMPLES = 1600          # 0.1 sec @ 16kHz (padding threshold)
CHUNK_SEC = 20.0            # chunk long audio to avoid OOM
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SEC)


# =========================
# WAV loading
# =========================
def load_wav_as_float32(path: Path) -> tuple[np.ndarray | None, str]:
    """
    Returns (audio, status).
    audio: float32 mono array @ 16kHz OR None if bad.
    status: reason for failure (for logging).
    """
    if not path.exists():
        return None, "MISSING_WAV"

    # Try soundfile first (best), fallback to scipy
    try:
        import soundfile as sf
        x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as e_sf:
        try:
            from scipy.io import wavfile
            sr, x = wavfile.read(str(path))
            # Convert int PCM to float32 [-1,1]
            if np.issubdtype(x.dtype, np.integer):
                x = x.astype(np.float32) / (np.iinfo(x.dtype).max + 1.0)
            else:
                x = x.astype(np.float32)
        except Exception:
            return None, f"READ_FAIL({type(e_sf).__name__})"

    if sr != SAMPLE_RATE:
        return None, f"WRONG_SR({sr})"

    if x is None:
        return None, "EMPTY_READ"

    # Convert to mono
    if isinstance(x, np.ndarray) and x.ndim > 1:
        x = x.mean(axis=1)

    x = np.asarray(x, dtype=np.float32)

    # Clean NaN/Inf
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # Super short / empty
    if x.size < 2:
        return None, f"TOO_SHORT({x.size})"

    return x, "OK"


# =========================
# Embedding
# =========================
@torch.no_grad()
def embed_audio_meanpool(
    x: np.ndarray,
    processor: Wav2Vec2Processor,
    model: Wav2Vec2Model,
    device: str,
) -> np.ndarray:
    """
    Mean pool last_hidden_state over time.
    Handles long audio by chunking and averaging chunk embeddings.
    Always returns L2-normalized float32 vector.
    """
    # chunk
    if len(x) <= CHUNK_SAMPLES:
        chunks = [x]
    else:
        chunks = [x[i:i + CHUNK_SAMPLES] for i in range(0, len(x), CHUNK_SAMPLES)]

    chunk_vecs: list[np.ndarray] = []
    for c in chunks:
        # pad chunk if needed
        if len(c) < MIN_SAMPLES:
            c = np.pad(c, (0, MIN_SAMPLES - len(c)), mode="constant")

        inputs = processor(c, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)

        out = model(input_values)  # (B, T, H)
        h = out.last_hidden_state

        v = h.mean(dim=1).squeeze(0)  # (H,)
        v = F.normalize(v, p=2, dim=0)
        chunk_vecs.append(v.detach().cpu().numpy().astype("float32"))

    v_all = np.mean(np.stack(chunk_vecs, axis=0), axis=0).astype("float32")
    v_all = v_all / (np.linalg.norm(v_all) + 1e-12)
    return v_all


def log_bad(video_id: str, reason: str) -> None:
    with open(BAD_LOG, "a") as f:
        f.write(f"{video_id}\t{reason}\n")


# =========================
# Main
# =========================
def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # reset bad log for clean run
    if BAD_LOG.exists():
        BAD_LOG.unlink()

    df = pd.read_csv(META_CSV)
    if "video_id" not in df.columns:
        raise SystemExit(f"Expected 'video_id' column in {META_CSV}, got: {list(df.columns)}")

    video_ids = df["video_id"].astype(str).tolist()

    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    hidden = model.config.hidden_size

    # We'll keep alignment by storing embeddings for only successful items
    # and writing an idmap matching embedding rows.
    X = np.zeros((len(video_ids), hidden), dtype="float32")
    kept_ids: list[str] = []

    miss_or_bad = 0

    for vid in tqdm(video_ids, desc="Embedding audio"):
        wav_path = WAV_DIR / f"{vid}.wav"

        x, status = load_wav_as_float32(wav_path)
        if x is None:
            miss_or_bad += 1
            log_bad(vid, status)
            continue

        # Pad very short audio so Wav2Vec2 conv layers won't crash
        if x.size < MIN_SAMPLES:
            x = np.pad(x, (0, MIN_SAMPLES - x.size), mode="constant")

        try:
            v = embed_audio_meanpool(x, processor, model, device)
        except RuntimeError as e:
            # Catch rare corner cases and keep going
            miss_or_bad += 1
            log_bad(vid, f"RUNTIME_ERROR({str(e).splitlines()[-1][:160]})")
            continue

        X[len(kept_ids)] = v
        kept_ids.append(vid)

    X = X[:len(kept_ids)]
    np.save(OUT_NPY, X)

    with open(OUT_IDMAP, "w") as f:
        json.dump(
            {
                "video_ids": kept_ids,
                "model": MODEL_NAME,
                "sample_rate": SAMPLE_RATE,
                "pooling": "mean_over_time_chunked",
                "chunk_sec": CHUNK_SEC,
                "min_samples_pad": MIN_SAMPLES,
            },
            f,
            indent=2,
        )

    print(f"\nSaved: {OUT_NPY}  shape={X.shape}")
    print(f"Saved: {OUT_IDMAP}")
    print(f"Bad/missing wavs: {miss_or_bad}")
    if BAD_LOG.exists():
        print(f"Bad log: {BAD_LOG}  (show with: tail -n 20 {BAD_LOG})")


if __name__ == "__main__":
    main()
