# Kokoro-82M Frozen Voice Cloning Pipeline

A highly optimized CPU-only voice cloning search pipeline for a frozen Kokoro-82M Text-to-Speech (TTS) model.

The objective is to clone a target speaker's voice using only 8 short audio reference clips and transcripts, without any fine-tuning or gradient calculations on the Kokoro model. The pipeline matches speaker embeddings using Resemblyzer and dynamically blends stock voice style tensors, then refines them using evolutionary search.

---

## Repository Structure

*   **`main.py`**: Pipeline entry point that orchestrates ranking, weight blending search, and evolution strategy optimization.
*   **`starter/`**
    *   **`synth.py`**: Model loading and TTS synthesis wrapper around the Kokoro pipeline.
    *   **`similarity.py`**: Evaluates speaker embedding matching using a cached Resemblyzer VoiceEncoder.
    *   **`blend.py`**: Implements vectorized stock voice ranking and Dirichlet sampling over the weight simplex.
    *   **`search.py`**: Implements evolutionary CMA-ES and random walk perturbation algorithms alongside stability checking.
    *   **`dims.py`**: Computes dimensional sensitivity analysis on the 256 style channels.
*   **`kokoro_assets/`**: Kokoro configuration, weights, and stock voice style tensors (shape: `(510, 1, 256)`).
*   **`reference/`**: Reference transcripts and audio clips of the target speaker.

---

## Core Optimizations

1.  **Fast CPU Speech Generation**: Bypasses PyTorch autograd allocation during iterations by wrapping speech synthesis inside `torch.inference_mode()`.
2.  **Stock Voice Caching**: Synthesizes and embeds the 54 stock voices once, caching their representations permanently in `stock_embeddings_cache.npy`. Subsequent runs load them instantly.
3.  **Short-Text Search Iterations**: Accelerates search iterations by synthesizing and comparing shortened text slices (first 4 words) during optimization loop updates, evaluating the full text sentence only once at the end.
4.  **Audio Stability Terms**: The search fitness function balances speaker similarity with a spectral flatness and silence ratio penalty, preventing search algorithms from drifting into degraded audio that fools the embedding.

---

## Setup & Running

Ensure you have **`espeak-ng`** installed and added to your system `PATH`.

### 1. Environment Verification
```powershell
$env:PATH += ";C:\Program Files\eSpeak NG"
python starter/setup_env.py --check
```

### 2. Run Voice Cloning
Run the entire voice cloning search pipeline (outputs the cloned `voice.pt` and prints the final similarity score):
```powershell
$env:PATH += ";C:\Program Files\eSpeak NG"
python main.py --reference_dir reference --texts_file reference/transcripts.txt --iterations 50 --output voice.pt
```

### 3. Run Block Sensitivity Analysis
Evaluate which blocks of the 256 style dimensions most strongly affect speech style and output a ranked list of block indices by sensitivity:
```powershell
$env:PATH += ";C:\Program Files\eSpeak NG"
python starter/dims.py --reference_dir reference --voice kokoro_assets/voices/af_heart.pt
```
