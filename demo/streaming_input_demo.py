"""Phase 0 PoC demo for input-streaming TTS via continuation chaining.

Feeds text chunks one-by-one to VoxCPM2 and concatenates the resulting audio.
Voice color is locked to a reference WAV (loli voice) so subsequent chunks
preserve timbre/prosody.

Usage:
    python demo/streaming_input_demo.py \\
        --ref /workspace/projects/voxcpm/out/loli/01_cute_basic.wav \\
        --output /workspace/projects/voxcpm/out/streaming_input.wav

See docs/streaming-input.md for the overall design.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

os.environ.setdefault("HF_HOME", "/workspace/data/voxcpm-cache")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from voxcpm import VoxCPM


# 8 chunks at natural clause boundaries. Simulates an LLM streaming its output
# token-by-token, with a small upstream buffer that flushes at commas/periods.
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref",
        default="/workspace/projects/voxcpm/out/loli/01_cute_basic.wav",
        help="Reference WAV for voice cloning (sticks across all chunks).",
    )
    parser.add_argument(
        "--output",
        default="/workspace/projects/voxcpm/out/streaming_input.wav",
        help="Concatenated output WAV path.",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=2.0,
        help="Diffusion CFG scale (lower=more natural, higher=closer to prompt).",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=10,
        help="Diffusion sampling steps per patch.",
    )
    parser.add_argument(
        "--model",
        default="openbmb/VoxCPM2",
        help="HF repo id or local path to the VoxCPM2 checkpoint.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.ref):
        sys.exit(f"reference WAV not found: {args.ref}")

    print(">> loading VoxCPM2 ...", flush=True)
    t0 = time.time()
    model = VoxCPM.from_pretrained(args.model, load_denoiser=False)
    sr = model.tts_model.sample_rate
    print(f">> loaded in {time.time() - t0:.1f}s  sr={sr}", flush=True)

    def text_stream():
        # In a real setup this would be an async generator pulled from an LLM.
        # Here we yield chunks with a small synthetic delay so the print log
        # shows the streaming timing.
        for chunk in DEFAULT_CHUNKS:
            print(f"   feeding: {chunk!r}", flush=True)
            yield chunk

    chunks_out = []
    start = time.time()
    first_audio_at = None
    for i, wav in enumerate(
        model.generate_streaming_input(
            text_stream(),
            reference_wav_path=args.ref,
            cfg_value=args.cfg,
            inference_timesteps=args.timesteps,
        )
    ):
        now = time.time() - start
        if first_audio_at is None:
            first_audio_at = now
            print(f"   TTFA  (first audio chunk): {now * 1000:.0f} ms", flush=True)
        else:
            print(f"   chunk {i}: t+{now * 1000:.0f} ms  ({len(wav) / sr:.2f} s audio)", flush=True)
        chunks_out.append(wav.astype(np.float32))

    total = time.time() - start
    if not chunks_out:
        sys.exit("no audio produced")
    full = np.concatenate(chunks_out)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, full, sr)
    print(
        f">> done. wall={total:.2f}s  audio={len(full) / sr:.2f}s "
        f"RTF={total / max(len(full) / sr, 1e-6):.3f} -> {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
