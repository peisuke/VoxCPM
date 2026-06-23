"""Compare per-patch speed of three code paths:
1. model.generate(text=...) — top-level VoxCPM.generate
2. tts_model._generate(...) — model-level internal _generate
3. tts_model._generate_with_prompt_cache(...) — what generate_streaming_input uses

If (3) is slower than (1)/(2), my streaming_input PoC has a real bottleneck
that's NOT just "more iters", but per-iter cost.
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/data/voxcpm-cache")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import soundfile as sf

from voxcpm import VoxCPM
from voxcpm.model.utils import next_and_close

REF = "/workspace/projects/voxcpm/out/loli/01_cute_basic.wav"
TXT = "お兄ちゃん、おかえりなさい。"

print(">> loading", flush=True)
model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
m = model.tts_model
sr = m.sample_rate

print(">> WARMUP: generate (top-level)", flush=True)
_ = model.generate(text=TXT, reference_wav_path=REF, cfg_value=2.0, inference_timesteps=10)

# Path A: top-level generate
t = time.time()
wav = model.generate(text=TXT, reference_wav_path=REF, cfg_value=2.0, inference_timesteps=10)
ta = time.time() - t
print(f"A: model.generate            {ta:.2f}s  audio={len(wav)/sr:.2f}s  RTF={ta/(len(wav)/sr):.3f}")

# Path B: tts_model._generate
t = time.time()
wav_t = next_and_close(m._generate(target_text=TXT, reference_wav_path=REF, cfg_value=2.0, inference_timesteps=10))
tb = time.time() - t
wb_dur = wav_t.shape[-1] / sr
print(f"B: tts_model._generate       {tb:.2f}s  audio={wb_dur:.2f}s  RTF={tb/wb_dur:.3f}")

# Path C: tts_model._generate_with_prompt_cache (what our streaming PoC uses)
cache = m.build_prompt_cache(reference_wav_path=REF)
t = time.time()
wav_t, _, _ = m.generate_with_prompt_cache(
    target_text=TXT,
    prompt_cache=cache,
    cfg_value=2.0,
    inference_timesteps=10,
)
tc = time.time() - t
wc_dur = wav_t.shape[-1] / sr
print(f"C: generate_with_prompt_cache {tc:.2f}s  audio={wc_dur:.2f}s  RTF={tc/wc_dur:.3f}")

# Print whether optimize was applied
print(f"\noptimize info: feat_encoder type = {type(m.feat_encoder).__name__}")
print(f"base_lm.forward_step type = {type(m.base_lm.forward_step).__name__}")
