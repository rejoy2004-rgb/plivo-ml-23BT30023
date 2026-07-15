import functools
import glob
import os
import numpy as np
import soundfile as sf

SR = 24000
_ENCODER = None

def load_encoder() -> any:
    global _ENCODER
    if _ENCODER is None:
        from resemblyzer import VoiceEncoder
        _ENCODER = VoiceEncoder("cpu", verbose=False)
    return _ENCODER

@functools.lru_cache(maxsize=None)
def _embed_file(path: str) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    from resemblyzer import preprocess_wav
    return load_encoder().embed_utterance(preprocess_wav(wav, source_sr=sr))

@functools.lru_cache(maxsize=1024)
def _embed_bytes(wav_bytes: bytes) -> np.ndarray:
    wav = np.frombuffer(wav_bytes, dtype=np.float32)
    from resemblyzer import preprocess_wav
    return load_encoder().embed_utterance(preprocess_wav(wav, source_sr=SR))

def embed(wav: str | np.ndarray) -> np.ndarray:
    if isinstance(wav, str):
        return _embed_file(wav)
    return _embed_bytes(wav.tobytes())

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray | float:
    a_arr = np.atleast_2d(a)
    b_arr = np.atleast_2d(b)
    a_norm = a_arr / (np.linalg.norm(a_arr, axis=1, keepdims=True) + 1e-9)
    b_norm = b_arr / (np.linalg.norm(b_arr, axis=1, keepdims=True) + 1e-9)
    sims = np.dot(a_norm, b_norm.T)
    if a.ndim == 1 and b.ndim == 1:
        return float(sims[0, 0])
    return np.squeeze(sims)

def target_embedding(reference_dir: str) -> np.ndarray:
    files = sorted(glob.glob(os.path.join(reference_dir, "*.wav")))
    if not files:
        raise FileNotFoundError()
    embs = np.stack([embed(f) for f in files])
    m = embs.mean(axis=0)
    return m / (np.linalg.norm(m) + 1e-9)

def similarity_to_target(wav: np.ndarray, target_emb: np.ndarray, sr: int = SR) -> float:
    return float(cosine_sim(embed(wav), target_emb))
