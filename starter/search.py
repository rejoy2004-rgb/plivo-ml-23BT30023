import numpy as np
import torch
import synth
import similarity as sim

def fitness(voice: torch.Tensor, target_emb: np.ndarray, texts: list[str]) -> float:
    scores = []
    for text in texts:
        try:
            wav, sr = synth.synthesize(text, voice)
            if len(wav) == 0: return -1.0
            s = sim.similarity_to_target(wav, target_emb, sr)
            p = np.abs(np.fft.rfft(wav)) ** 2 + 1e-9
            flatness = np.exp(np.mean(np.log(p))) / np.mean(p)
            sil = np.mean(np.abs(wav) < 1e-3)
            clip = np.mean(np.abs(wav) >= 0.99)
            eng = np.mean(wav ** 2)
            penalty = 0.0
            if flatness > 0.1: penalty += 2.0 * (flatness - 0.1)
            if sil > 0.4: penalty += 2.0 * (sil - 0.4)
            if clip > 0.01: penalty += 5.0 * clip
            if eng < 1e-5: penalty += 2.0
            elif eng > 0.1: penalty += 2.0 * (eng - 0.1)
            scores.append(s - penalty)
        except Exception: return -1.0
    return float(np.mean(scores))

_CACHE = {}
def get_fitness(voice: torch.Tensor, target_emb: np.ndarray, texts: list[str]) -> float:
    h = hash(voice.cpu().numpy().tobytes())
    if h not in _CACHE: _CACHE[h] = fitness(voice, target_emb, texts)
    return _CACHE[h]

class PerturbationSearch:
    def __init__(self, initial_tensor: torch.Tensor, target_embedding: np.ndarray, texts: list[str]):
        self.best_tensor, self.target_embedding, self.texts = initial_tensor.clone(), target_embedding, texts
        self.best_score = get_fitness(self.best_tensor, target_embedding, self.texts)
    def step(self, tensor: torch.Tensor, sigma: float) -> torch.Tensor:
        cand = tensor + torch.from_numpy(sigma * np.random.randn(*tensor.shape)).to(tensor.dtype)
        score = get_fitness(cand, self.target_embedding, self.texts)
        if score > self.best_score: self.best_tensor, self.best_score = cand, score
        return cand

class CMAESSearch:
    def __init__(self, initial_tensor: torch.Tensor, target_embedding: np.ndarray, texts: list[str], pop_size: int = 4):
        self.mean, self.sigma, self.pop_size, self.target_embedding, self.texts = initial_tensor.clone(), 0.05, pop_size, target_embedding, texts
        self.best_tensor, self.best_score = self.mean.clone(), get_fitness(self.mean, target_embedding, texts)
    def step(self, sigma: float = None) -> tuple[torch.Tensor, float]:
        if sigma is not None: self.sigma = sigma
        cands = [self.mean + self.sigma * torch.randn_like(self.mean) for _ in range(self.pop_size)]
        scores = [get_fitness(c, self.target_embedding, self.texts) for c in cands]
        idx = np.argsort(scores)[::-1]
        if scores[idx[0]] > self.best_score: self.best_tensor, self.best_score = cands[idx[0]].clone(), scores[idx[0]]
        self.mean = torch.stack([cands[i] for i in idx[:max(1, self.pop_size // 2)]]).mean(dim=0)
        return self.best_tensor, self.best_score

def run_search(initial_tensor: torch.Tensor, target_embedding: np.ndarray, texts: list[str], iterations: int) -> tuple[torch.Tensor, float, list[tuple[int, float]]]:
    searcher = CMAESSearch(initial_tensor, target_embedding, texts)
    hist = []
    for i in range(1, iterations + 1):
        sigma = 0.05 if i <= int(iterations * 0.3) else (0.02 if i <= int(iterations * 0.7) else 0.005)
        best_t, best_s = searcher.step(sigma)
        hist.append((i, best_s))
    return best_t, best_s, hist
