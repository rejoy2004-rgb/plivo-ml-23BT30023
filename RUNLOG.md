# Voice Cloning Optimization Runlog

This log tracks the progression of experimental runs, configuration changes, and research hypotheses.

---

### Run 1: Nearest Stock Voice Baseline
*   **Method**: Zero-shot matching of target speaker embedding against all 54 stock voices.
*   **Config**: N/A
*   **Results**:
    *   **Best Stock Voice**: `bf_lily`
    *   **Target Similarity**: `0.4728`
    *   **Vibe/Vocal Quality**: Natural, but voice does not sound like the target speaker.

### Run 2: Dirichlet Blend Weights Search
*   **Method**: Convex combination of the top 5 stock voices using random Dirichlet sampling on the weight simplex.
*   **Config**: `n_samples = 30`, evaluation on short text `"The quick brown fox."`
*   **Results**:
    *   **Best Weight Vector**: `[0.342, 0.118, 0.287, 0.155, 0.098]`
    *   **Target Similarity**: `0.5312`
    *   **Vibe/Vocal Quality**: Blending successfully captures target timbre but lacks speaker-specific formants.

### Run 3: Separable CMA-ES in Full 130k-Dimensional Space
*   **Method**: Diagonal CMA-ES search in raw style parameter space (`510 * 256 = 130,560` parameters) with hierarchical noise schedule.
*   **Config**: `iterations = 50`, `pop_size = 4`, `sigma_init = 0.05`.
    *   **Fitness weights**: `w_sim = 1.0`, `w_flatness = 2.0`, `w_silence = 2.0`, `w_clipping = 5.0`, `w_energy = 2.0`.
*   **Results**:
    *   **Target Similarity**: `0.6019`
    *   **Vibe/Vocal Quality**: Exhibited minor digital artifacts, robotic formants, and unnatural sibilance.
    *   **Hypothesis**: High-dimensional unconstrained search easily wanders off the manifold of realistic voices to game the Resemblyzer embedding.

### Run 4: Subspace CMA-ES in PCA Basis (Final Optimized Setup)
*   **Method**: Separable diagonal CMA-ES inside a 15-dimensional PCA subspace computed from the 54 stock voice tensors, utilizing range clamping and sentence rotation.
*   **Config**: `K = 15` components, `iterations = 50`, `pop_size = 4`, `sigma = 0.05 -> 0.02 -> 0.005`, `seed = 42`.
    *   **Fitness formulation**: 
        $$\text{fitness} = \text{similarity} - 2.0 \max(0, \text{flatness} - 0.1) - 2.0 \max(0, \text{silence} - 0.4) - 5.0 \times \text{clipping} - \text{energy\_penalty}$$
*   **Results**:
    *   **Target Similarity**: `0.5698`
    *   **Vibe/Vocal Quality**: **Highly natural, smooth, and intelligible**. Clamping coordinates in PCA space restricted parameters to the manifold of human voice distributions, completely preventing speech degradation and robotic noise.
