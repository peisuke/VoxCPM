"""Engine sanity check: drive StreamingInputSession with ONE chunk.

If the streaming engine itself is correct, this should produce valid
Japanese roughly equivalent to ``model.generate(text=...)``. If the output
is broken even with a single chunk, the bug is in the engine (not in the
chunk-interleaving learning task).

Usage:
    python demo/streaming_session_1chunk.py \\
        --ref /workspace/projects/voxcpm/out/loli/01_cute_basic.wav \\
        --output /workspace/projects/voxcpm/out/streaming_session_1chunk.wav
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


# Same total text as the 8-chunk demo, but fed in ONE go.
ONE_CHUNK = ("お兄ちゃん、おかえりなさい。今日もお仕事、おつかれさまっ。"
             "ねえねえ、聞いてほしいことが、あるんだあ。"
             "ちょっとだけ、時間もらってもいい?")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ref", default="/workspace/projects/voxcpm/out/loli/01_cute_basic.wav")
    p.add_argument("--output", default="/workspace/projects/voxcpm/out/streaming_session_1chunk.wav")
    p.add_argument("--cfg", type=float, default=2.0)
    p.add_argument("--timesteps", type=int, default=10)
    p.add_argument("--max-patches", type=int, default=200)
    args = p.parse_args()

    print(">> loading", flush=True)
    t0 = time.time()
    wrapper = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
    sr = wrapper.tts_model.sample_rate
    print(f">> loaded in {time.time() - t0:.1f}s", flush=True)

    # ----- A. Reference: model.generate() = the trained pathway -----
    print(">> A. baseline via model.generate() (=正規パス、必ず動くはず)", flush=True)
    t = time.time()
    wav_ref = wrapper.generate(
        text=ONE_CHUNK, reference_wav_path=args.ref,
        cfg_value=args.cfg, inference_timesteps=args.timesteps,
    )
    print(f"   wall={time.time() - t:.2f}s  audio={len(wav_ref) / sr:.2f}s", flush=True)
    ref_out = Path(args.output).with_name("session_1chunk_REF.wav")
    sf.write(ref_out, wav_ref.astype(np.float32), sr)

    # ----- B. Engine: StreamingInputSession with 1 chunk -----
    print(">> B. via StreamingInputSession (=Phase 1-A エンジン、1チャンク)", flush=True)
    sess = wrapper.create_streaming_session(
        reference_wav_path=args.ref,
        cfg_value=args.cfg,
        inference_timesteps=args.timesteps,
    )
    t = time.time()
    sess.feed_text(ONE_CHUNK)
    chunks_out = []
    for wav in sess.flush_audio(max_patches=args.max_patches, min_patches=2):
        w = wav.float().numpy() if hasattr(wav, "float") else np.asarray(wav)
        chunks_out.append(w.reshape(-1).astype(np.float32))
    sess.close()
    wav_engine = np.concatenate(chunks_out) if chunks_out else np.array([], dtype=np.float32)
    print(f"   wall={time.time() - t:.2f}s  audio={len(wav_engine) / sr:.2f}s", flush=True)
    sf.write(args.output, wav_engine, sr)
    print(f">> wrote\n   A: {ref_out}\n   B: {args.output}", flush=True)


if __name__ == "__main__":
    main()
