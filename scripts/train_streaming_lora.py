"""Phase 1-C: LoRA fine-tune VoxCPM2 TSLM on interleaved text/audio chunks.

Reads a chunked manifest produced by ``scripts/prepare_streaming_manifest.py``
(JSONL with ``audio``, ``text``, ``chunks: [{text, start, end}]`` fields) and
fine-tunes the TSLM via LoRA so the model learns to consume text chunk by
chunk and emit audio chunk by chunk.

LoRA targets:
- ``base_lm`` + ``residual_lm`` (LM-side) only.
- LocDiT and AudioVAE are frozen.

Sequence layout per sample (VibeVoice-style per-chunk boundary markers):

    [text_chunk_0 ids, audio_start_id, audio_feats_0..., audio_end_id,
     text_chunk_1 ids, audio_start_id, audio_feats_1..., audio_end_id,
     ...
     text_chunk_K ids, audio_start_id, audio_feats_K..., audio_end_id]

Stop labels are 1 at the last audio patch of each chunk (= the position right
before its audio_end_id), 0 elsewhere. Loss is computed on audio positions
only (loss_mask = audio_mask).

Usage:
    python scripts/train_streaming_lora.py \\
        --manifest /workspace/data/reazonspeech-small/manifest.train.jsonl \\
        --val-manifest /workspace/data/reazonspeech-small/manifest.val.jsonl \\
        --output /workspace/data/voxcpm-streaming-lora \\
        --iters 500 --lr 1e-4 --rank 16 --alpha 32
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault("HF_HOME", "/workspace/data/hf-cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from voxcpm.model.voxcpm2 import VoxCPM2Model, LoRAConfig
from voxcpm.model.utils import get_dtype


# ------------------------------------------------------------------ #
# Dataset
# ------------------------------------------------------------------ #


class StreamingChunkDataset(Dataset):
    """Reads a JSONL manifest of {audio, text, chunks: [{text, start, end}]}
    and returns dict containing raw waveform + chunk info.

    Heavy work (audio encoding + tokenization + sequence assembly) is done
    by the collate fn so we can amortize torch tensor work and pin to the
    model's device.
    """

    def __init__(self, manifest_path: str, sample_rate: int = 16_000, max_duration_s: float = 20.0):
        self.records: List[dict] = []
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                dur = r.get("duration", 0.0) or 0.0
                if dur > max_duration_s:
                    continue
                # Drop samples whose chunks have any missing timestamps.
                chunks = r.get("chunks") or []
                if not chunks or any(c.get("start") is None or c.get("end") is None for c in chunks):
                    continue
                # Drop tiny ones where chunks won't fit even 1 patch.
                if dur < 0.5:
                    continue
                self.records.append(r)
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        wav, sr = sf.read(r["audio"], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        if sr != self.sample_rate:
            raise ValueError(f"unexpected sample rate {sr} for {r['audio']}")
        return {
            "wav": wav,
            "chunks": r["chunks"],
            "duration": r.get("duration", float(len(wav)) / sr),
            "id": r["id"],
        }


# ------------------------------------------------------------------ #
# Sequence builder
# ------------------------------------------------------------------ #


def _quantize_time_to_patch(t_sec: float, audio_vae_fps: float, patch_size: int) -> int:
    """Snap a time in seconds to a patch index (1 patch = patch_size VAE frames)."""
    frames = t_sec * audio_vae_fps  # float
    patch = round(frames / patch_size)
    return max(0, int(patch))


def build_streaming_sequence(
    sample: dict,
    model: VoxCPM2Model,
    audio_vae_fps: float,
    patch_size: int,
    audio_start_id: int,
    audio_end_id: int,
):
    """Construct the interleaved (text_chunk → audio_chunk → ...) training
    sequence for one sample.

    Returns dict with text_tokens, text_mask, audio_feats, audio_mask,
    loss_mask, labels — all 1D / 3D tensors on CPU. The batch collate is
    a separate step.
    """
    device = "cpu"
    chunks = sample["chunks"]
    wav = torch.from_numpy(sample["wav"])

    # Encode full waveform once, then slice into patches per chunk.
    with torch.no_grad():
        wav_in = wav.unsqueeze(0).unsqueeze(0).to(model.device)  # [1, 1, T]
        hop_len = model.audio_vae.hop_length
        plen = hop_len * patch_size
        if wav_in.size(-1) % plen != 0:
            pad = plen - (wav_in.size(-1) % plen)
            wav_in = F.pad(wav_in, (0, pad))
        z = model.audio_vae.encode(wav_in, model.audio_vae.sample_rate)  # [1, D, T_vae]
        feat = z.transpose(1, 2).squeeze(0).cpu()  # [T_vae, D]
    # Trim trailing partial patch
    T_vae = feat.shape[0]
    T_patch = T_vae // patch_size
    feat = feat[: T_patch * patch_size]
    # Reshape to patches: [T_patch, P, D]
    feat = feat.view(T_patch, patch_size, feat.shape[-1])

    # Tokenizer is callable; returns list[int] of text token ids.
    tokenize = model.text_tokenizer
    P = patch_size
    D = feat.shape[-1]

    # Assemble interleaved sequence.
    text_ids: List[int] = []          # int32 tokens; for audio positions we put 0
    audio_pieces: List[torch.Tensor] = []  # contributes one P,D row per position
    text_mask_l: List[int] = []
    audio_mask_l: List[int] = []
    loss_mask_l: List[int] = []

    def push_text_token(tok: int):
        text_ids.append(tok)
        audio_pieces.append(torch.zeros(P, D, dtype=feat.dtype))
        text_mask_l.append(1)
        audio_mask_l.append(0)
        loss_mask_l.append(0)

    def push_audio_patch(patch: torch.Tensor):
        text_ids.append(0)
        audio_pieces.append(patch)
        text_mask_l.append(0)
        audio_mask_l.append(1)
        loss_mask_l.append(1)

    # VibeVoice-style per-chunk boundary tokens. Each chunk produces:
    #     [text tokens] <SOA> [audio patches] <EOA>
    # The <EOA> after every audio block gives the model an explicit "this
    # chunk is done, next text chunk is coming" signal, matching what the
    # inference loop in streaming.py pushes when stop fires.
    for i, ch in enumerate(chunks):
        toks = tokenize(ch["text"])
        for t in toks:
            push_text_token(int(t))
        push_text_token(audio_start_id)
        s = _quantize_time_to_patch(float(ch["start"]), audio_vae_fps, patch_size)
        e = _quantize_time_to_patch(float(ch["end"]), audio_vae_fps, patch_size)
        e = min(max(e, s + 1), T_patch)
        for p_idx in range(s, e):
            push_audio_patch(feat[p_idx])
        push_text_token(audio_end_id)

    text_tokens = torch.tensor(text_ids, dtype=torch.int32)
    audio_feats = torch.stack(audio_pieces, dim=0)  # [T, P, D]
    text_mask = torch.tensor(text_mask_l, dtype=torch.int32)
    audio_mask = torch.tensor(audio_mask_l, dtype=torch.int32)
    loss_mask = torch.tensor(loss_mask_l, dtype=torch.int32)
    labels = torch.zeros(text_tokens.size(0), dtype=torch.int32)
    # Mark the last audio patch of EACH chunk as a stop target. A position is
    # "end of a chunk" when the next position in the sequence is not also an
    # audio patch — i.e. when chunk-text resumes (or the closing audio_end_id
    # / pad lands). Marking only the last audio of the whole sequence taught
    # the model "never emit stop mid-stream", which at inference time made
    # every chunk run to max_patches and produce the looped audio we heard.
    audio_idx = (audio_mask == 1).nonzero(as_tuple=False).squeeze(-1)
    for i in range(audio_idx.numel()):
        pos = int(audio_idx[i].item())
        is_last_in_chunk = (
            i == audio_idx.numel() - 1
            or int(audio_idx[i + 1].item()) != pos + 1
        )
        if is_last_in_chunk:
            labels[pos] = 1

    return {
        "text_tokens": text_tokens,
        "audio_feats": audio_feats,
        "text_mask": text_mask,
        "audio_mask": audio_mask,
        "loss_mask": loss_mask,
        "labels": labels,
    }


def collate_streaming(batch_dicts: List[dict], pad_audio_d: int, patch_size: int) -> dict:
    """Pad-collate a list of sequences produced by build_streaming_sequence."""
    seqs = batch_dicts
    max_len = max(s["text_tokens"].size(0) for s in seqs)

    def pad1d(t: torch.Tensor, pad_val: int) -> torch.Tensor:
        if t.size(0) == max_len:
            return t
        return F.pad(t, (0, max_len - t.size(0)), value=pad_val)

    def pad3d(t: torch.Tensor) -> torch.Tensor:
        if t.size(0) == max_len:
            return t
        return F.pad(t, (0, 0, 0, 0, 0, max_len - t.size(0)))

    out = {
        "text_tokens": torch.stack([pad1d(s["text_tokens"], 0) for s in seqs], dim=0).long(),
        "audio_feats": torch.stack([pad3d(s["audio_feats"]) for s in seqs], dim=0),
        "text_mask": torch.stack([pad1d(s["text_mask"], 0) for s in seqs], dim=0),
        "audio_mask": torch.stack([pad1d(s["audio_mask"], 0) for s in seqs], dim=0),
        "loss_mask": torch.stack([pad1d(s["loss_mask"], 0) for s in seqs], dim=0),
        "position_ids": torch.stack(
            [F.pad(torch.arange(s["text_tokens"].size(0)),
                   (0, max_len - s["text_tokens"].size(0))) for s in seqs], dim=0
        ).long(),
        "labels": torch.stack([pad1d(s["labels"], 0) for s in seqs], dim=0).long(),
    }
    return out


# ------------------------------------------------------------------ #
# Training
# ------------------------------------------------------------------ #


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--val-manifest", default="")
    p.add_argument("--model", default="openbmb/VoxCPM2")
    p.add_argument("--output", required=True, help="Where to save LoRA checkpoints.")
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--val-every", type=int, default=100)
    p.add_argument("--max-duration", type=float, default=15.0,
                   help="Drop samples longer than this many seconds (KV cap).")
    args = p.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- model ----------
    print(">> loading model", flush=True)
    from voxcpm import VoxCPM
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
    model = wrapper.tts_model
    model.train()
    device = model.device
    dtype = get_dtype(model.config.dtype)
    print(f">> device={device} dtype={dtype}", flush=True)

    # Freeze everything; only LoRA params train.
    n_trainable = 0
    n_total = 0
    for name, param in model.named_parameters():
        n_total += param.numel()
        if "lora_" in name:
            param.requires_grad = True
            n_trainable += param.numel()
        else:
            param.requires_grad = False
    print(f">> trainable params: {n_trainable / 1e6:.2f}M / {n_total / 1e6:.1f}M "
          f"({100 * n_trainable / n_total:.3f}%)", flush=True)

    # ---------- data ----------
    print(">> loading datasets", flush=True)
    train_ds = StreamingChunkDataset(args.manifest, max_duration_s=args.max_duration)
    print(f"   train: {len(train_ds)} samples", flush=True)
    val_ds = None
    if args.val_manifest and os.path.exists(args.val_manifest):
        val_ds = StreamingChunkDataset(args.val_manifest, max_duration_s=args.max_duration)
        print(f"   val:   {len(val_ds)} samples", flush=True)

    # audio_vae_fps and patch_size for time→patch mapping
    audio_vae_fps = float(model.audio_vae.sample_rate) / float(model.audio_vae.hop_length)
    patch_size = model.patch_size
    feat_dim = model.audio_vae.latent_dim
    print(f">> audio_vae_fps={audio_vae_fps:.2f}  patch_size={patch_size}  feat_dim={feat_dim}", flush=True)

    def make_loader(ds: Dataset, shuffle: bool) -> DataLoader:
        def _collate_inner(batch):
            seqs = [build_streaming_sequence(
                s, model, audio_vae_fps, patch_size,
                audio_start_id=model.audio_start_token,
                audio_end_id=model.audio_end_token if hasattr(model, "audio_end_token") else 102,
            ) for s in batch]
            return collate_streaming(seqs, feat_dim, patch_size)
        return DataLoader(
            ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=0,
            collate_fn=_collate_inner,
        )

    train_loader = make_loader(train_ds, shuffle=True)
    val_loader = make_loader(val_ds, shuffle=False) if val_ds else None

    # ---------- optimizer ----------
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-2)

    def lr_lambda(step: int) -> float:
        if step < args.warmup:
            return float(step + 1) / float(args.warmup)
        progress = (step - args.warmup) / max(1, args.iters - args.warmup)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    # ---------- train ----------
    print(">> starting training", flush=True)
    step = 0
    accum_diff = 0.0
    accum_stop = 0.0
    t0 = time.time()
    losses_log = []
    data_iter = iter(train_loader)

    while step < args.iters:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        out = model(
            text_tokens=batch["text_tokens"],
            text_mask=batch["text_mask"],
            audio_feats=batch["audio_feats"],
            audio_mask=batch["audio_mask"],
            loss_mask=batch["loss_mask"],
            position_ids=batch["position_ids"],
            labels=batch["labels"],
            progress=float(step) / max(1, args.iters),
        )
        loss = out["loss/diff"] + out["loss/stop"]
        (loss / args.grad_accum).backward()

        accum_diff += float(out["loss/diff"].detach().item())
        accum_stop += float(out["loss/stop"].detach().item())

        if (step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)

        step += 1
        if step % args.log_every == 0:
            dt = time.time() - t0
            avg_diff = accum_diff / args.log_every
            avg_stop = accum_stop / args.log_every
            cur_lr = sched.get_last_lr()[0]
            print(f"  step {step:5d}/{args.iters}  "
                  f"diff={avg_diff:.4f}  stop={avg_stop:.4f}  "
                  f"lr={cur_lr:.2e}  ({step / dt:.2f} it/s)", flush=True)
            losses_log.append({"step": step, "diff": avg_diff, "stop": avg_stop, "lr": cur_lr})
            accum_diff = 0.0
            accum_stop = 0.0

        if val_loader is not None and step % args.val_every == 0:
            model.eval()
            with torch.no_grad():
                v_diff = v_stop = 0.0; v_n = 0
                for vb in val_loader:
                    o = model(
                        text_tokens=vb["text_tokens"], text_mask=vb["text_mask"],
                        audio_feats=vb["audio_feats"], audio_mask=vb["audio_mask"],
                        loss_mask=vb["loss_mask"], position_ids=vb["position_ids"],
                        labels=vb["labels"],
                    )
                    v_diff += float(o["loss/diff"].item())
                    v_stop += float(o["loss/stop"].item())
                    v_n += 1
                    if v_n >= 20: break
                print(f"  [val] step {step}  diff={v_diff/v_n:.4f}  stop={v_stop/v_n:.4f}", flush=True)
            model.train()

        if step % args.save_every == 0 or step == args.iters:
            ckpt_path = out_dir / f"lora_step{step}.pt"
            lora_state = wrapper.get_lora_state_dict()
            torch.save({
                "step": step,
                "lora_config": {
                    "enable_lm": True, "enable_dit": False, "enable_proj": False,
                    "r": args.rank, "alpha": args.alpha,
                },
                "state_dict": lora_state,
            }, ckpt_path)
            (out_dir / "losses.jsonl").write_text(
                "".join(json.dumps(x) + "\n" for x in losses_log)
            )
            print(f"  saved {ckpt_path}", flush=True)

    print(">> done", flush=True)


if __name__ == "__main__":
    main()
