import numpy as np
import torch
import synth
import similarity as sim

def fitness(voice: torch.Tensor, target_emb: np.ndarray, texts: list[str]) -> float:
    scores = []
    for text in texts:
        try:
            wav, sr = synth.synthesize(text, voice)
            if len(wav) == 0:
                return -1.0
            s = sim.similarity_to_target(wav, target_emb, sr)
            fft_val = np.abs(np.fft.rfft(wav))
            power = fft_val ** 2 + 1e-9
            flatness = np.exp(np.mean(np.log(power))) / np.mean(power)
            silence = np.mean(np.abs(wav) < 1e-3)
            penalty = 0.0
            if flatness > 0.1:
                penalty += 2.0 * (flatness - 0.1)
            if silence > 0.4:
                penalty += 2.0 * (silence - 0.4)
            energy = np.mean(wav ** 2)
            if energy < 1e-5 or energy > 0.5:
                penalty += 1.0
            scores.append(s - penalty)
        except Exception:
            return -1.0
    return float(np.mean(scores))

class PerturbationSearch:
    def __init__(self, initial_tensor: torch.Tensor, target_embedding: np.ndarray, texts: list[str]):
        self.best_tensor = initial_tensor.clone()
        self.target_embedding = target_embedding
        self.texts = texts
        self.best_score = fitness(self.best_tensor, target_embedding, self.texts)

    def step(self, tensor: torch.Tensor, sigma: float) -> torch.Tensor:
        noise = sigma * np.random.randn(*tensor.shape)
        cand = tensor + torch.from_numpy(noise).to(tensor.dtype)
        score = fitness(cand, self.target_embedding, self.texts)
        if score > self.best_score:
            self.best_tensor, self.best_score = cand, score
        return cand

class CMAESSearch:
    def __init__(self, initial_tensor: torch.Tensor, target_embedding: np.ndarray, texts: list[str], pop_size: int = 4):
        self.mean = initial_tensor.clone()
        self.sigma = 0.03
        self.pop_size = pop_size
        self.target_embedding = target_embedding
        self.texts = texts
        self.best_tensor = self.mean.clone()
        self.best_score = fitness(self.best_tensor, target_embedding, texts)

    def step(self) -> tuple[torch.Tensor, float]:
        cands = [self.mean + self.sigma * torch.randn_like(self.mean) for _ in range(self.pop_size)]
        scores = [fitness(c, self.target_embedding, self.texts) for c in cands]
        indices = np.argsort(scores)[::-1]
        if scores[indices[0]] > self.best_score:
            self.best_tensor, self.best_score = cands[indices[0]].clone(), scores[indices[0]]
        selected = [cands[idx] for idx in indices[:max(1, self.pop_size // 2)]]
        self.mean = torch.stack(selected).mean(dim=0)
        return self.best_tensor, self.best_score

def run_search(initial_tensor: torch.Tensor, target_embedding: np.ndarray, texts: list[str], iterations: int) -> tuple[torch.Tensor, float, list[tuple[int, float]]]:
    searcher = CMAESSearch(initial_tensor, target_embedding, texts)
    history = []
    for i in range(1, iterations + 1):
        best_t, best_s = searcher.step()
        history.append((i, best_s))
    return best_t, best_s, history
