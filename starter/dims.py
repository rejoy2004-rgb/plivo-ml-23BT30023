import argparse
import sys
import numpy as np
import torch
import synth
import similarity as sim
import search

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference_dir", required=True)
    ap.add_argument("--voice", required=True)
    args = ap.parse_args()
    target = sim.target_embedding(args.reference_dir)
    base_voice = synth.load_voice_tensor(args.voice)
    texts = ["The quick brown fox jumps over the lazy dog."]
    base_score = search.fitness(base_voice, target, texts)
    block_size = 16
    n_blocks = 256 // block_size
    deltas = []
    for i in range(n_blocks):
        perturbed = base_voice.clone()
        start = i * block_size
        end = start + block_size
        noise = 0.5 * torch.randn(perturbed.shape[0], perturbed.shape[1], block_size)
        perturbed[:, :, start:end] += noise
        score = search.fitness(perturbed, target, texts)
        delta = abs(score - base_score)
        deltas.append((i, delta))
    deltas.sort(key=lambda x: x[1], reverse=True)
    ranked = [idx for idx, _ in deltas]
    sys.stdout.write(f"{ranked}\n")

if __name__ == "__main__":
    main()
