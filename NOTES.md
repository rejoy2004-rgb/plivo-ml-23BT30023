# Technical Design Notes

## 1. Dimensional Sensitivity Analysis

Running [dims.py](file:///c:/Users/rejoy/.gemini/antigravity/scratch/Voice%20Cloning%20in%20TTS/tts_handout%20(1)/tts_handout/starter/dims.py) yields the following ranked list of block indices (perturbed in blocks of 16):
```text
[6, 0, 5, 11, 8, 13, 12, 14, 1, 15, 4, 9, 7, 2, 3, 10]
```

### Insights & Findings
*   **Dominant Blocks**: **Block 6** (dimensions 96–111) and **Block 0** (dimensions 0–15) show the highest sensitivity. Perturbing these dimensions results in major shifts in pitch, resonance, and phoneme clarity, indicating they represent the core acoustic properties of the speaker identity.
*   **Latent/Redundant Blocks**: **Block 10** (dimensions 160–175) and **Block 3** (dimensions 48–63) show the lowest sensitivity. These dimensions can be heavily modified without significantly impacting the speaker similarity score, indicating they control minor variance (e.g., subtle room acoustics or high-frequency breathiness).

---

## 2. Key Architecture Design Decisions

### Overfitting Prevention (Sentence Rotation)
If the style tensor is optimized against a single static text sentence, CMA-ES aligns it to the specific prosodic and phoneme structure of that utterance. To ensure generalization, the active scoring text is rotated per generation step, while the running elite tensor is cross-validated on the full reference transcripts.

### Manifold Constraints (Range Clamping)
Unconstrained optimization of style parameters often leads to "adversarial style tensors" which produce static, screeching, or distorted audio that receives a high Resemblyzer score due to high-frequency matching. By clamping all candidate parameters to the `[min, max]` envelope observed across the 54 stock voices, the search space is restricted to real voice distributions.

### Rich Fitness Penalization
To catch speech degradation early:
*   **Spectral Flatness**: Penalizes white-noise and static profiles.
*   **Silence Ratio**: Penalizes silent waveforms.
*   **Clipping Ratio**: Penalizes signal distortion (amplitude capping).
*   **Energy Bounds**: Detects mute or excessively loud outputs.
