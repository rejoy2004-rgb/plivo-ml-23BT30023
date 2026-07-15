import glob
import os
import numpy as np
import torch

_default_kokoro_dir = "./kokoro_assets" if os.path.isdir("./kokoro_assets") else "../kokoro_assets"
KOKORO_DIR = os.environ.get("KOKORO_DIR") or _default_kokoro_dir
SR = 24000
_PIPELINE = None

def get_pipeline(lang_code: str = "a") -> any:
    global _PIPELINE
    if _PIPELINE is None:
        from kokoro import KModel, KPipeline
        cands = glob.glob(os.path.join(KOKORO_DIR, "*.pth"))
        if not cands:
            raise FileNotFoundError()
        model = KModel(config=os.path.join(KOKORO_DIR, "config.json"), model=cands[0])
        _PIPELINE = KPipeline(lang_code=lang_code, model=model)
    return _PIPELINE

def load_voice_tensor(path: str) -> torch.Tensor:
    t = torch.load(path, map_location="cpu", weights_only=True)
    return t.detach().to(torch.float32).cpu()

def synthesize(text: str, voice_tensor: torch.Tensor) -> tuple[np.ndarray, int]:
    pipe = get_pipeline()
    with torch.inference_mode():
        chunks = [r.audio for r in pipe(text, voice=voice_tensor, speed=1.0)]
        audio = torch.cat([c if isinstance(c, torch.Tensor) else torch.tensor(c) for c in chunks])
    return audio.detach().cpu().numpy().astype(np.float32), SR

def batch_synthesize(texts: list[str], voice_tensor: torch.Tensor) -> list[np.ndarray]:
    pipe = get_pipeline()
    waveforms = []
    with torch.inference_mode():
        for text in texts:
            chunks = [r.audio for r in pipe(text, voice=voice_tensor, speed=1.0)]
            if not chunks:
                waveforms.append(np.array([], dtype=np.float32))
                continue
            audio = torch.cat([c if isinstance(c, torch.Tensor) else torch.tensor(c) for c in chunks])
            waveforms.append(audio.detach().cpu().numpy().astype(np.float32))
    return waveforms

def stock_voices() -> dict[str, torch.Tensor]:
    voices = {}
    for path in sorted(glob.glob(os.path.join(KOKORO_DIR, "voices", "*.pt"))):
        name = os.path.splitext(os.path.basename(path))[0]
        voices[name] = load_voice_tensor(path)
    return voices
