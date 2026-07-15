import glob
import os
import numpy as np
import torch
import synth
import similarity as sim
from scipy.spatial.distance import cdist

def rank_stock_voices(target_embedding: np.ndarray, voice_dir: str) -> list[tuple[str, float]]:
    paths = sorted(glob.glob(os.path.join(voice_dir, "*.pt")))
    names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    cache_path = os.path.join(voice_dir, "stock_embeddings_cache.npy")
    if os.path.exists(cache_path):
        embs = np.load(cache_path)
    else:
        tensors = [synth.load_voice_tensor(p) for p in paths]
        text = "The quick brown fox."
        wavs = [synth.synthesize(text, t)[0] for t in tensors]
        embs = np.stack([sim.embed(w) for w in wavs])
        try:
            np.save(cache_path, embs)
        except Exception:
            pass
    scores = 1.0 - cdist(embs, target_embedding[None, :], metric="cosine")[:, 0]
    return sorted(zip(names, scores), key=lambda x: x[1], reverse=True)

def convex_blend(voices: list[torch.Tensor], weights: list[float] | np.ndarray) -> torch.Tensor:
    stacked = torch.stack(voices)
    w_tensor = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device)
    return (stacked * w_tensor.view(-1, 1, 1, 1)).sum(dim=0)

def search_blend_weights(top_k_voices: list[torch.Tensor], target_embedding: np.ndarray, n_samples: int) -> tuple[np.ndarray, float]:
    weights = np.random.dirichlet(np.ones(len(top_k_voices)), size=n_samples)
    cands = [convex_blend(top_k_voices, w) for w in weights]
    text = "The quick brown fox."
    wavs = [synth.synthesize(text, c)[0] for c in cands]
    embs = np.stack([sim.embed(w) for w in wavs])
    scores = 1.0 - cdist(embs, target_embedding[None, :], metric="cosine")[:, 0]
    best_idx = np.argmax(scores)
    return weights[best_idx], float(scores[best_idx])
