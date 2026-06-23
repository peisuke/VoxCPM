"""Phase 1-C inference: streaming session with the LoRA-trained TSLM.

Compares against the Phase 1-A baseline (no LoRA) on the same script and
reference voice. Generates two WAVs:
    streaming_session.wav     # Phase 1-A (no LoRA) — re-uses existing file
    streaming_session_lora.wav # Phase 1-C (with LoRA from training)

Usage:
    python demo/streaming_lora_demo.py \\
        --lora /workspace/data/voxcpm-streaming-lora/lora_step500.pt \\
        --ref  /workspace/projects/voxcpm/out/loli/01_cute_basic.wav \\
        --output /workspace/projects/voxcpm/out/streaming_session_lora.wav
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

os.environ.setdefault("HF_HOME", "/workspace/data/hf-cache")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from voxcpm import VoxCPM
from voxcpm.model.voxcpm2 import LoRAConfig


DEFAULT_CHUNKS = [
    "お兄ちゃん、",
    "おかえりなさい。",
    "今日もお仕事、",
    "おつかれさまっ。",
    "ねえねえ、",
    "聞いてほしいことが、",
    "あるんだあ。",
    "ちょっとだけ、時間もらってもいい?",
]


def load_lora_state(model, ckpt_path: str) -> None:
    """Load LoRA weights from a .pt produced by train_streaming_lora.py."""
    ckpt = torch.load(ckpt_path, map_location=model.device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    model_params = dict(model.named_parameters())
    key_mapping = {k.replace("._orig_mod.", "."): k for k in model_params if "._orig_mod." in k}
    loaded = skipped = 0
    for key, value in state_dict.items():
        target_key = key if key in model_params else key_mapping.get(key)
        if target_key:
            model_params[target_key].data.copy_(value.to(model.device))
            loaded += 1
        else:
            skipped += 1
    print(f">> LoRA: loaded {loaded} tensors, skipped {skipped}", flush=True)


def run_session(wrapper, args, lora_label: str) -> tuple[np.ndarray, float]:
    sess = wrapper.create_streaming_session(
        reference_wav_path=args.ref,
        cfg_value=args.cfg,
        inference_timesteps=args.timesteps,
    )
    all_wav = []
    start = time.time()
    ttfa = None
    try:
        for idx, chunk in enumerate(DEFAULT_CHUNKS):
            sess.feed_text(chunk)
            is_last = idx == len(DEFAULT_CHUNKS) - 1
            max_patches = 30 if is_last else args.patches_per_chunk
            for wav in sess.flush_audio(max_patches=max_patches, min_patches=0):
                w = wav.float().numpy() if hasattr(wav, "float") else np.asarray(wav)
                w = w.reshape(-1).astype(np.float32)
                all_wav.append(w)
                if ttfa is None:
                    ttfa = (time.time() - start) * 1000
                    print(f"  [{lora_label}] TTFA: {ttfa:.0f} ms", flush=True)
    finally:
        sess.close()
    full = np.concatenate(all_wav) if all_wav else np.array([], dtype=np.float32)
    wall = time.time() - start
    return full, wall


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lora", required=True)
    p.add_argument("--ref", default="/workspace/projects/voxcpm/out/loli/01_cute_basic.wav")
    p.add_argument("--output", default="/workspace/projects/voxcpm/out/streaming_session_lora.wav")
    p.add_argument("--cfg", type=float, default=2.0)
    p.add_argument("--timesteps", type=int, default=10)
    p.add_argument("--patches-per-chunk", type=int, default=14)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--model", default="openbmb/VoxCPM2")
    args = p.parse_args()

    print(">> loading model with LoRA enabled (uninitialized A/B; will load from ckpt)", flush=True)
    t0 = time.time()
    wrapper = VoxCPM.from_pretrained(
        args.model,
        load_denoiser=False,
        lora_config=LoRAConfig(
            enable_lm=True,
            enable_dit=False,
            enable_proj=False,
            r=args.rank,
            alpha=args.alpha,
        ),
    )
    print(f">> base model loaded in {time.time() - t0:.1f}s", flush=True)

    sr = wrapper.tts_model.sample_rate

    # ---------- A. Warm up with random LoRA (=baseline no-LoRA equivalent) ----------
    # Initial LoRA is identity (A is kaiming init, B is zeros → effective output = base only),
    # so this captures Phase 1-A behavior without loading our checkpoint.
    print(">> warmup (LoRA un-initialized = base behavior)", flush=True)
    _ = wrapper.generate(text="ウォームアップ。", reference_wav_path=args.ref,
                         cfg_value=args.cfg, inference_timesteps=args.timesteps)

    wav_base, wall_base = run_session(wrapper, args, "BASE")
    print(f"  [BASE] wall={wall_base:.2f}s  audio={len(wav_base) / sr:.2f}s "
          f"RTF={wall_base / max(len(wav_base) / sr, 1e-6):.3f}", flush=True)
    base_out = Path(args.output).with_name(Path(args.output).stem + "_base.wav")
    sf.write(base_out, wav_base, sr)
    print(f"  wrote {base_out}", flush=True)

    # ---------- B. Load trained LoRA ----------
    print(">> loading trained LoRA from", args.lora, flush=True)
    load_lora_state(wrapper.tts_model, args.lora)

    wav_lora, wall_lora = run_session(wrapper, args, "LORA")
    print(f"  [LORA] wall={wall_lora:.2f}s  audio={len(wav_lora) / sr:.2f}s "
          f"RTF={wall_lora / max(len(wav_lora) / sr, 1e-6):.3f}", flush=True)
    sf.write(args.output, wav_lora, sr)
    print(f"  wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
