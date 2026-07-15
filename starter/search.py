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
            sil, clip, eng = np.mean(np.abs(wav) < 1e-3), np.mean(np.abs(wav) >= 0.99), np.mean(wav ** 2)
            penalty = 2.0 * max(0.0, flatness - 0.1) + 2.0 * max(0.0, sil - 0.4) + 5.0 * clip
            penalty += 2.0 if eng < 1e-5 else (2.0 * max(0.0, eng - 0.1))
            scores.append(s - penalty)
        except Exception: return -1.0
    return float(np.mean(scores))

_CACHE = {}
def get_fitness(v: torch.Tensor, t: np.ndarray, txts: list[str]) -> float:
    h = hash(v.cpu().numpy().tobytes())
    if h not in _CACHE: _CACHE[h] = fitness(v, t, txts)
    return _CACHE[h]

class PCAProjector:
    def __init__(self, stock: dict[str, torch.Tensor], K: int = 15):
        self.K, X = K, torch.stack(list(stock.values())).view(len(stock), -1)
        self.mean = X.mean(dim=0)
        _, self.S, V = torch.linalg.svd(X - self.mean, full_matrices=False)
        self.V = V[:K]
        z = torch.matmul(X - self.mean, self.V.t())
        self.z_min, self.z_max = z.min(dim=0)[0], z.max(dim=0)[0]
    def project(self, z: torch.Tensor) -> torch.Tensor: return (self.mean + torch.matmul(z, self.V)).view(510, 1, 256)
    def encode(self, voice: torch.Tensor) -> torch.Tensor: return torch.matmul(voice.view(-1) - self.mean, self.V.t())

class PerturbationSearch:
    def __init__(self, init_t: torch.Tensor, target: np.ndarray, texts: list[str]):
        self.best_tensor, self.target, self.texts = init_t.clone(), target, texts
        self.best_score = get_fitness(init_t, target, texts)
    def step(self, tensor: torch.Tensor, sigma: float) -> torch.Tensor:
        cand = tensor + torch.from_numpy(sigma * np.random.randn(*tensor.shape)).to(tensor.dtype)
        score = get_fitness(cand, self.target, self.texts)
        if score > self.best_score: self.best_tensor, self.best_score = cand, score
        return cand

class CMAESSearch:
    def __init__(self, initial_tensor: torch.Tensor, target_embedding: np.ndarray, texts: list[str], proj: PCAProjector, pop_size: int = 4):
        self.proj, self.pop_size, self.target_embedding, self.texts = proj, pop_size, target_embedding, texts
        self.z, self.sigma, self.C_diag = proj.encode(initial_tensor), 0.05, torch.ones(proj.K)
        self.best_tensor, self.best_score, self.step_count = initial_tensor.clone(), get_fitness(initial_tensor, target_embedding, texts), 0
    def step(self, sigma: float = None) -> tuple[torch.Tensor, float]:
        if sigma is not None: self.sigma = sigma
        active_text = [self.texts[self.step_count % len(self.texts)]]
        self.step_count += 1
        z_cands = [torch.clamp(self.z + self.sigma * self.C_diag * torch.randn_like(self.z), self.proj.z_min, self.proj.z_max) for _ in range(self.pop_size)]
        scores = [get_fitness(self.proj.project(zc), self.target_embedding, active_text) for zc in z_cands]
        idx = np.argsort(scores)[::-1]
        selected_z = [z_cands[i] for i in idx[:max(1, self.pop_size // 2)]]
        old_z = self.z.clone()
        self.z = torch.stack(selected_z).mean(dim=0)
        self.C_diag = 0.9 * self.C_diag + 0.1 * torch.stack([z - old_z for z in selected_z]).std(dim=0).clamp(0.01, 2.0)
        best_cand = self.proj.project(self.z)
        score_full = get_fitness(best_cand, self.target_embedding, self.texts)
        if score_full > self.best_score: self.best_tensor, self.best_score = best_cand, score_full
        return self.best_tensor, self.best_score

def run_search(initial_tensor: torch.Tensor, target_embedding: np.ndarray, texts: list[str], iterations: int) -> tuple[torch.Tensor, float, list[tuple[int, float]]]:
    proj = PCAProjector(synth.stock_voices())
    searcher = CMAESSearch(initial_tensor, target_embedding, texts, proj)
    hist, plateau, prev_best = [], 0, searcher.best_score
    for i in range(1, iterations + 1):
        sigma = (0.05 if i <= int(iterations * 0.3) else (0.02 if i <= int(iterations * 0.7) else 0.005)) * (0.5 if plateau >= 8 else 1.0)
        if plateau >= 8: plateau = 0
        best_t, best_s = searcher.step(sigma)
        hist.append((i, best_s))
        plateau = 0 if best_s > prev_best else (plateau + 1)
        prev_best = max(prev_best, best_s)
        if i % 10 == 0: torch.save(best_t, f"checkpoint_step_{i}.pt")
    return best_t, best_s, hist
