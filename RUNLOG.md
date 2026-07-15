# Voice Cloning Optimization Runlog

This log tracks the progression of experimental runs and optimizations performed to clone the target speaker voice under CPU and time constraints.

---

### Run 1: Nearest Stock Voice Baseline
*   **Method**: Zero-shot matching of target speaker embedding against all 54 stock voices.
*   **Config**: N/A
*   **Results**:
    *   **Best Stock Voice**: `bf_lily`
    *   **Target Similarity**: `0.4728`
    *   **Audio Quality**: High, natural, but vocal characteristics do not closely match the target speaker.

### Run 2: Dirichlet Blend Weights Search
*   **Method**: Convex combination of the top 5 stock voices using random Dirichlet sampling on the weight simplex.
*   **Config**: `n_samples = 30`, evaluation on short text `"The quick brown fox."`
*   **Results**:
    *   **Best Weight Vector**: `[0.342, 0.118, 0.287, 0.155, 0.098]`
    *   **Target Similarity**: `0.5312`
    *   **Audio Quality**: High, blending characteristics successfully capture target resonance.

### Run 3: Isotropic CMA-ES Search
*   **Method**: Local CMA-ES search using a fixed global step size (`sigma = 0.03`).
*   **Config**: `iterations = 5`, `pop_size = 4`, evaluation on full transcript sentence.
*   **Results**:
    *   **Target Similarity**: `0.6019`
    *   **Audio Quality**: Intelligible, but started exhibiting minor robotic robotic artifacts.

### Run 4: Optimized Separable CMA-ES (Final Setup)
*   **Method**: True diagonal CMA-ES with standard deviation updates (`C_diag`) per-dimension, range clamping constraints, sentence rotation, and plateau-based noise scheduling.
*   **Config**: `iterations = 50`, `pop_size = 4`, `sigma_init = 0.05`, `plateau_patience = 8`.
*   **Results**:
    *   **Target Similarity**: `0.5493`
    *   **Audio Quality**: **Excellent and natural**. Range clamping bounds successfully prevented the optimizer from drifting into high-score adversarial static. Sentence rotation ensured phonetic robustness.
