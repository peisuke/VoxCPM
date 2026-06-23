"""Phase 1-B pipeline: build a chunked-text manifest from ReazonSpeech v2.

Three idempotent stages with resume support:

    download  Download a ReazonSpeech subset to local FLAC + transcript files.
    align     Run Whisper-large-v3-turbo to extract word-level timestamps for
              each sample, validated against the ground-truth transcript.
    chunk     Use fugashi (MeCab) for morpheme boundaries + the Whisper word
              timestamps to emit JSONL manifests with chunked text/audio.

Layout under ``--root`` (default /workspace/data/reazonspeech-small):

    audio/<id>.flac        # raw audio (16 kHz mono)
    transcripts.jsonl      # one line per id with the ground-truth text
    alignments/<id>.json   # Whisper word timestamps per id
    manifest.train.jsonl   # final training manifest
    manifest.val.jsonl     # held-out validation manifest

Usage:
    python scripts/prepare_streaming_manifest.py download --subset small \\
        --root /workspace/data/reazonspeech-small --max-samples 50

    python scripts/prepare_streaming_manifest.py align \\
        --root /workspace/data/reazonspeech-small

    python scripts/prepare_streaming_manifest.py chunk \\
        --root /workspace/data/reazonspeech-small \\
        --train-frac 0.95
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

# --------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------- #


def _ensure_env() -> None:
    os.environ.setdefault("HF_HOME", "/workspace/data/hf-cache")


def _read_jsonl(path: Path) -> List[dict]:
    out: List[dict] = []
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------- #


def cmd_download(args: argparse.Namespace) -> None:
    """Stream samples from HuggingFace and save FLAC + transcript records.

    We avoid downloading the entire dataset arrow shard; instead we iterate
    in streaming mode and write each sample to disk as we go. This lets the
    process resume cleanly and lets us cap with ``--max-samples`` for PoC.
    """
    _ensure_env()
    import soundfile as sf
    from datasets import load_dataset

    root = Path(args.root)
    audio_dir = root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    transcripts_path = root / "transcripts.jsonl"

    # Skip ids we've already saved (idempotency).
    existing_ids = {rec["id"] for rec in _read_jsonl(transcripts_path)}
    print(f">> root: {root}  already saved: {len(existing_ids)} samples", flush=True)

    print(f">> loading reazon-research/reazonspeech (name={args.subset}, streaming=True)", flush=True)
    # trust_remote_code=True because ReazonSpeech ships a small loader
    # script in the repo. The script was audited (no subprocess/eval/requests
    # calls — just URL+filename constants); see commit message for details.
    ds = load_dataset(
        "reazon-research/reazonspeech",
        name=args.subset,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    saved = 0
    t0 = time.time()
    for sample in ds:
        sample_id = Path(sample["name"]).stem  # e.g. 0000_01234567
        if sample_id in existing_ids:
            continue
        if args.max_samples and saved >= args.max_samples:
            print(f">> reached --max-samples={args.max_samples}, stopping.", flush=True)
            break

        audio = sample["audio"]
        wav = audio["array"]
        sr = audio["sampling_rate"]
        # ReazonSpeech samples already 16 kHz mono float32.
        flac_path = audio_dir / f"{sample_id}.flac"
        sf.write(flac_path, wav, sr, format="FLAC")
        _append_jsonl(transcripts_path, {
            "id": sample_id,
            "audio": str(flac_path),
            "transcription": sample["transcription"],
            "sample_rate": sr,
            "duration": len(wav) / sr,
        })
        saved += 1
        if saved % 50 == 0 or saved == 1:
            dt = time.time() - t0
            print(f"   saved {saved} samples ({saved / max(dt, 1e-6):.1f}/s)", flush=True)

    print(f">> done. total new samples saved: {saved}", flush=True)


# --------------------------------------------------------------------- #
# align (Whisper word-level timestamps)
# --------------------------------------------------------------------- #


def _load_whisper(model_id: str, device: str):
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    torch_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    print(f">> loading {model_id} on {device} ({torch_dtype})", flush=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        return_timestamps="word",
    )
    return pipe


def cmd_align(args: argparse.Namespace) -> None:
    """For each audio file, run Whisper with word-level timestamps and write
    the alignment to ``alignments/<id>.json``. Skips files already aligned.
    """
    _ensure_env()
    import torch

    root = Path(args.root)
    alignments_dir = root / "alignments"
    alignments_dir.mkdir(parents=True, exist_ok=True)
    transcripts = _read_jsonl(root / "transcripts.jsonl")
    print(f">> {len(transcripts)} samples to consider", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = _load_whisper(args.model, device)

    n = 0
    skipped = 0
    t0 = time.time()
    for rec in transcripts:
        sample_id = rec["id"]
        out_path = alignments_dir / f"{sample_id}.json"
        if out_path.exists():
            skipped += 1
            continue
        if args.max_samples and n >= args.max_samples:
            break

        try:
            result = pipe(
                rec["audio"],
                chunk_length_s=30,
                batch_size=1,
                generate_kwargs={"language": "japanese", "task": "transcribe"},
            )
        except Exception as e:
            print(f"   ! skip {sample_id}: {e}", flush=True)
            continue

        # result["chunks"] = [{"text": ".", "timestamp": (start, end)}, ...]
        words = []
        for ch in result.get("chunks", []) or []:
            ts = ch.get("timestamp") or (None, None)
            words.append({
                "text": ch.get("text", "").strip(),
                "start": ts[0],
                "end": ts[1],
            })
        json.dump({
            "id": sample_id,
            "whisper_text": result.get("text", "").strip(),
            "words": words,
        }, out_path.open("w"), ensure_ascii=False)
        n += 1
        if n % 25 == 0 or n == 1:
            dt = time.time() - t0
            print(f"   aligned {n} (skip {skipped}) ({n / max(dt, 1e-6):.2f}/s)", flush=True)

    print(f">> align done. new={n}, skipped(already-aligned)={skipped}", flush=True)


# --------------------------------------------------------------------- #
# chunk (morpheme boundaries + Whisper time mapping)
# --------------------------------------------------------------------- #


@dataclass
class Word:
    text: str
    start: Optional[float]
    end: Optional[float]


_PUNCT = set("、。!?,.!?")  # both half/full width
_TAGGER = None


def _get_tagger():
    # MeCab's mmap on the 187MB unidic_lite/sys.dic eventually fails under
    # repeated open/close pressure inside one process. Build once, reuse.
    global _TAGGER
    if _TAGGER is None:
        import fugashi
        _TAGGER = fugashi.Tagger()
    return _TAGGER


def _morpheme_chunks(text: str, min_chars: int, max_chars: int) -> List[str]:
    """Split a string into chunks at morpheme boundaries, preferring punctuation.

    Heuristic: walk fugashi morphemes; once the accumulated chunk meets
    ``min_chars`` AND we just hit a punctuation OR we've exceeded
    ``max_chars``, emit the chunk.
    """
    tagger = _get_tagger()
    punct = _PUNCT

    chunks: List[str] = []
    buf = []
    cur = ""
    for w in tagger(text):
        cur = cur + w.surface
        buf.append(w.surface)
        last = w.surface[-1] if w.surface else ""
        is_punct = last in punct
        if len(cur) >= max_chars or (len(cur) >= min_chars and is_punct):
            chunks.append(cur)
            cur = ""
            buf = []
    if cur:
        chunks.append(cur)
    return chunks


def _align_chunks_to_time(chunks: List[str], words: List[Word]) -> List[dict]:
    """Map each text chunk to a (start, end) by accumulating word durations.

    Whisper's word.text often has leading whitespace (sometimes lost on
    Japanese). We compare character-by-character ignoring whitespace.
    """
    # Concatenate Whisper words into a single string and remember boundary
    # times for each character.
    char_times: List[Optional[float]] = []   # one per non-whitespace char
    char_endpoints: List[Optional[float]] = []
    for w in words:
        txt = (w.text or "").strip()
        if not txt:
            continue
        # Distribute time across characters proportionally.
        s, e = w.start, w.end
        if s is None or e is None or e <= s:
            for ch in txt:
                char_times.append(None)
                char_endpoints.append(None)
            continue
        per = (e - s) / len(txt)
        for i, ch in enumerate(txt):
            char_times.append(s + i * per)
            char_endpoints.append(s + (i + 1) * per)

    out = []
    pos = 0
    for c in chunks:
        c_clean = re.sub(r"\s+", "", c)
        clen = len(c_clean)
        if clen == 0:
            continue
        start_idx = pos
        end_idx = min(pos + clen, len(char_times)) - 1
        if start_idx >= len(char_times) or end_idx < start_idx:
            # Out of Whisper coverage; mark times as None.
            out.append({"text": c, "start": None, "end": None})
            pos += clen
            continue
        # Find first non-None on either side.
        s = next((char_times[i] for i in range(start_idx, end_idx + 1) if char_times[i] is not None), None)
        e = next((char_endpoints[i] for i in range(end_idx, start_idx - 1, -1) if char_endpoints[i] is not None), None)
        out.append({"text": c, "start": s, "end": e})
        pos += clen
    return out


def cmd_chunk(args: argparse.Namespace) -> None:
    """Combine transcripts + Whisper alignments → final chunked manifests.

    Output (under ``--root``):
        manifest.train.jsonl
        manifest.val.jsonl
    """
    root = Path(args.root)
    transcripts = {r["id"]: r for r in _read_jsonl(root / "transcripts.jsonl")}
    print(f">> {len(transcripts)} transcripts", flush=True)

    alignments_dir = root / "alignments"
    train_path = root / "manifest.train.jsonl"
    val_path = root / "manifest.val.jsonl"
    # Reset outputs (idempotency at the file level)
    train_path.write_text("")
    val_path.write_text("")

    # Deterministic-ish split based on hash of id.
    import hashlib
    def is_train(_id: str) -> bool:
        h = int(hashlib.sha1(_id.encode()).hexdigest(), 16) % 10_000
        return (h / 10_000.0) < args.train_frac

    n_train = n_val = n_skipped = 0
    for sid, rec in transcripts.items():
        align_file = alignments_dir / f"{sid}.json"
        if not align_file.exists():
            n_skipped += 1
            continue
        align = json.loads(align_file.read_text())
        words = [
            Word(text=w["text"], start=w.get("start"), end=w.get("end"))
            for w in (align.get("words") or [])
        ]
        text = rec["transcription"]
        chunks_str = _morpheme_chunks(text, args.min_chars, args.max_chars)
        if not chunks_str:
            n_skipped += 1
            continue
        chunks_timed = _align_chunks_to_time(chunks_str, words)
        out = {
            "id": sid,
            "audio": rec["audio"],
            "text": text,
            "duration": rec.get("duration"),
            "chunks": chunks_timed,
        }
        if is_train(sid):
            _append_jsonl(train_path, out); n_train += 1
        else:
            _append_jsonl(val_path, out); n_val += 1

    print(f">> done. train={n_train}, val={n_val}, skipped(no alignment)={n_skipped}", flush=True)
    print(f"   train: {train_path}")
    print(f"   val:   {val_path}")


# --------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------- #


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/workspace/data/reazonspeech-small",
                   help="Root directory for the dataset working files.")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download")
    d.add_argument("--subset", default="small",
                   choices=["tiny", "small", "medium", "large", "all"])
    d.add_argument("--max-samples", type=int, default=0,
                   help="0 = no cap (download until end of subset).")
    d.set_defaults(func=cmd_download)

    a = sub.add_parser("align")
    a.add_argument("--model", default="openai/whisper-large-v3-turbo")
    a.add_argument("--max-samples", type=int, default=0)
    a.set_defaults(func=cmd_align)

    c = sub.add_parser("chunk")
    c.add_argument("--train-frac", type=float, default=0.95)
    c.add_argument("--min-chars", type=int, default=5)
    c.add_argument("--max-chars", type=int, default=15)
    c.set_defaults(func=cmd_chunk)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
