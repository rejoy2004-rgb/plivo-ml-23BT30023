import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "starter"))
import torch
import synth
import similarity as sim
import blend
import search

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference_dir", required=True)
    ap.add_argument("--texts_file", required=True)
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--output", default="voice.pt")
    args = ap.parse_args()
    target = sim.target_embedding(args.reference_dir)
    with open(args.texts_file, "r") as f:
        texts = [line.strip() for line in f if line.strip()]
    ranked = blend.rank_stock_voices(target, "kokoro_assets/voices" if os.path.isdir("kokoro_assets/voices") else "starter/kokoro_assets/voices")
    top_names = [name for name, _ in ranked[:5]]
    stock = synth.stock_voices()
    top_voices = [stock[name] for name in top_names]
    best_weights, _ = blend.search_blend_weights(top_voices, target, 30)
    initial_blend = blend.convex_blend(top_voices, best_weights)
    search_texts = [" ".join(t.split()[:4]) for t in texts[:1]]
    best_tensor, _, _ = search.run_search(initial_blend, target, search_texts, args.iterations)
    final_score = search.fitness(best_tensor, target, texts[:1])
    torch.save(best_tensor, args.output)
    print(final_score)

if __name__ == "__main__":
    main()
