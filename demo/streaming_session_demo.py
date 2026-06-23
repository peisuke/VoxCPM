"""Phase 1-A demo: drive a StreamingInputSession with chunked text input.

Compared to ``streaming_input_demo.py`` (Phase 0), this maintains the TSLM /
RALM KV caches across chunks, so text additions are O(new tokens) instead of
O(full prefix). The AudioVAE streaming decoder is also kept open across the
whole session.

Expectation:
- TTFA (time-to-first-audio) << Phase 0 (~ a couple of patch generations only).
- Per-chunk latency stays roughly constant as the conversation grows.

Usage:
    python demo/streaming_session_demo.py \\
        --ref /workspace/projects/voxcpm/out/loli/01_cute_basic.wav \\
        --output /workspace/projects/voxcpm/out/streaming_session.wav
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

os.environ.setdefault("HF_HOME", "/workspace/data/hf-cache")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from voxcpm import VoxCPM


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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ref", default="/workspace/projects/voxcpm/out/loli/01_cute_basic.wav")
    p.add_argument("--output", default="/workspace/projects/voxcpm/out/streaming_session.wav")
    p.add_argument("--cfg", type=float, default=2.0)
    p.add_argument("--timesteps", type=int, default=10)
    p.add_argument("--patches-per-chunk", type=int, default=14,
                   help="Patches to flush after each text feed. 14 patches @ 6.25Hz ≈ 2.2s audio.")
    p.add_argument("--model", default="openbmb/VoxCPM2")
    args = p.parse_args()

    if not os.path.exists(args.ref):
        sys.exit(f"reference WAV not found: {args.ref}")

    print(">> loading", flush=True)
    t0 = time.time()
    model = VoxCPM.from_pretrained(args.model, load_denoiser=False)
    sr = model.tts_model.sample_rate
    print(f">> loaded in {time.time() - t0:.1f}s  sr={sr}", flush=True)

    # First end-to-end via the old single-shot API to warm up CUDA kernels and torch.compile.
    print(">> warmup (single-shot)", flush=True)
    _ = model.generate(text="ウォームアップ。", reference_wav_path=args.ref,
                       cfg_value=args.cfg, inference_timesteps=args.timesteps)

    print(">> opening streaming session", flush=True)
    sess = model.create_streaming_session(
        reference_wav_path=args.ref,
        cfg_value=args.cfg,
        inference_timesteps=args.timesteps,
    )

    all_wav = []
    overall_start = time.time()
    ttfa = None
    try:
        for idx, chunk in enumerate(DEFAULT_CHUNKS):
            t_feed_start = time.time()
            sess.feed_text(chunk)
            feed_dt = (time.time() - t_feed_start) * 1000
            print(f"  feed_text {idx}: {chunk!r}  ({feed_dt:.0f} ms for tokens)", flush=True)

            t_audio_start = time.time()
            chunk_audio = []
            is_last = idx == len(DEFAULT_CHUNKS) - 1
            # In Phase 1-A the stop predictor may not fire reliably on the
            # interleaved layout (training fixes this in Phase 1-C). Cap the
            # last chunk to a generous-but-bounded value so the demo finishes.
            max_patches = 30 if is_last else args.patches_per_chunk
            min_patches = 2 if is_last else 0
            for wav in sess.flush_audio(max_patches=max_patches, min_patches=min_patches):
                w = wav.float().numpy() if hasattr(wav, "float") else np.asarray(wav)
                # flush_audio yields shape [batch=1, samples]; flatten to 1D for concat.
                w = w.reshape(-1).astype(np.float32)
                chunk_audio.append(w)
                if ttfa is None:
                    ttfa = (time.time() - overall_start) * 1000
                    print(f"  TTFA: {ttfa:.0f} ms", flush=True)
            audio_dt = time.time() - t_audio_start
            if chunk_audio:
                wav_chunk = np.concatenate(chunk_audio)
                dur = len(wav_chunk) / sr
                print(f"  flush_audio {idx}: {audio_dt:.2f}s wall, {dur:.2f}s audio  RTF={audio_dt/max(dur,1e-6):.3f}",
                      flush=True)
                all_wav.append(wav_chunk)
            else:
                print(f"  flush_audio {idx}: 0 patches", flush=True)
    finally:
        sess.close()

    if not all_wav:
        sys.exit("no audio produced")

    full = np.concatenate(all_wav)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, full, sr)
    total = time.time() - overall_start
    print(f">> done. wall={total:.2f}s  audio={len(full)/sr:.2f}s  "
          f"RTF={total/max(len(full)/sr,1e-6):.3f} -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
