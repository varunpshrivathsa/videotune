from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[2]

META_CSV = PROJECT_ROOT / "data" / "processed" / "metadata.csv"
AUDIO_DIR = PROJECT_ROOT / "data" / "audio"
OUT_EMB = PROJECT_ROOT / "data" / "embeddings" / "audio_emb_full.npy"

SR = 16000
MAX_SEC = 12.0  # speed/robustness cap; set None if you want full length (not recommended)


def load_wav(path: Path) -> torch.Tensor:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)

    # If stereo, average channels
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Resample if needed (should already be 16k from your ffmpeg step)
    if sr != SR:
        import torchaudio
        wav = torch.from_numpy(audio)
        wav = torchaudio.functional.resample(wav, sr, SR)
    else:
        wav = torch.from_numpy(audio)

    if MAX_SEC is not None:
        max_len = int(SR * MAX_SEC)
        if wav.numel() > max_len:
            wav = wav[:max_len]

    return wav

@torch.no_grad()
def embed_one(model, processor, wav: torch.Tensor, device: str) -> np.ndarray:
    inputs = processor(wav.numpy(), sampling_rate=SR, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs).last_hidden_state  # (B, T, H)
    emb = out.mean(dim=1).squeeze(0)         # (H,)
    emb = torch.nn.functional.normalize(emb, dim=0)
    return emb.detach().cpu().numpy().astype("float32")

def main() -> None:
    df = pd.read_csv(META_CSV)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device).eval()

    X = np.zeros((len(df), 768), dtype="float32")
    missing = 0

    for i, row in enumerate(tqdm(df.itertuples(index=False), total=len(df))):
        vid = str(getattr(row, "video_id"))
        wav_path = AUDIO_DIR / f"{vid}.wav"
        if not wav_path.exists():
            missing += 1
            continue

        wav = load_wav(wav_path)
        X[i] = embed_one(model, processor, wav, device)

    OUT_EMB.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_EMB, X)

    print(f"Saved: {OUT_EMB}  shape={X.shape}  missing_audio={missing}")

if __name__ == "__main__":
    main()
