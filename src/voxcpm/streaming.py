"""Phase 1-A: true KV-cache continuation for input-streaming TTS.

Maintains TSLM and RALM KV caches across feed_text() and flush_audio() calls
so that incremental text additions cost O(new tokens) instead of O(full prefix).
The AudioVAE streaming decoder is kept open across the session for smooth
patch-by-patch waveform emission.

See ``docs/streaming-input.md`` for the overall design. This module implements
the inference engine only; quality on chunked-text inputs is addressed by
LoRA fine-tuning in Phase 1-C, since VoxCPM2 was originally trained on
fully-formed text-then-audio sequences.

Example:

    from voxcpm import VoxCPM

    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
    with model.create_streaming_session(reference_wav_path="ref.wav") as sess:
        sess.feed_text("お兄ちゃん、")
        for wav in sess.flush_audio(max_patches=20):
            ...
        sess.feed_text("おかえりなさい。")
        for wav in sess.flush_audio():
            ...
"""
from __future__ import annotations

from typing import Generator, List, Optional

import torch
from einops import rearrange

from .model.utils import get_dtype
from .model.voxcpm2 import VoxCPM2Model


class StreamingInputSession:
    """Stateful TTS session that interleaves text feeding and audio generation.

    Holds open the model's KV caches plus the AudioVAE streaming decoder so
    every successive call (feed_text / flush_audio) advances state incrementally.
    """

    def __init__(
        self,
        tts_model: VoxCPM2Model,
        reference_wav_path: Optional[str] = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        streaming_prefix_len: int = 4,
        max_kv_length: Optional[int] = None,
    ) -> None:
        self.m = tts_model
        self.device = tts_model.device
        self.dtype = get_dtype(tts_model.config.dtype)
        self.cfg_value = cfg_value
        self.inference_timesteps = inference_timesteps
        self.streaming_prefix_len = streaming_prefix_len

        # Allow re-sizing KV cache if a previous session was tight on space.
        if max_kv_length is not None:
            self.m.base_lm.setup_cache(1, max_kv_length, self.device, self.dtype)
            self.m.residual_lm.setup_cache(1, max_kv_length, self.device, self.dtype)

        # Reset existing caches.
        self.m.base_lm.kv_cache.current_length = 0
        self.m.base_lm.kv_cache.kv_cache.zero_()
        self.m.residual_lm.kv_cache.current_length = 0
        self.m.residual_lm.kv_cache.kv_cache.zero_()

        # Hidden states maintained across calls.
        self.lm_hidden: Optional[torch.Tensor] = None       # [1, h_lm]
        self.residual_hidden: Optional[torch.Tensor] = None  # [1, h_res]

        # LocDiT condition (the last clean audio patch latent).
        self.prefix_feat_cond: Optional[torch.Tensor] = None  # [1, p, d]

        # Generated patch latents, kept short for AudioVAE smoothing only.
        self.pred_feat_seq: List[torch.Tensor] = []  # list of [1, 1, p, d]

        # Whether <SOA> has been pushed (=text→audio transition has happened).
        self._audio_started = False

        # AudioVAE streaming decoder kept open across flush_audio calls.
        self._vae_ctx = self.m.audio_vae.streaming_decode()
        self._vae_dec = self._vae_ctx.__enter__()
        self._closed = False

        # Prefill the reference audio (if any). This populates the KV caches
        # and sets lm_hidden / residual_hidden / prefix_feat_cond.
        if reference_wav_path is not None:
            self._prefill_reference(reference_wav_path)
        else:
            # Zero-shot: nothing to prefill, audio prefix starts from zero.
            P = self.m.patch_size
            D = self.m.audio_vae.latent_dim
            self.prefix_feat_cond = torch.zeros((1, P, D), device=self.device, dtype=self.dtype)

    # ------------------------------------------------------------------ #
    # Prefill
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def _prefill_reference(self, reference_wav_path: str) -> None:
        """Encode reference audio and run a single prefill to seed KV caches."""
        # _encode_wav returns a CPU tensor; _make_ref_prefix cats it with
        # zero patches that must live on the same device, so build the prefix
        # on CPU and move the result to the model device afterwards.
        ref_feat = self.m._encode_wav(reference_wav_path, padding_mode="right")
        ref_tokens, ref_feats, ref_t_mask, ref_a_mask = self.m._make_ref_prefix(
            ref_feat, ref_feat.device
        )

        text_token = ref_tokens.unsqueeze(0).to(self.device)
        feat = ref_feats.unsqueeze(0).to(self.device).to(self.dtype)
        text_mask = ref_t_mask.unsqueeze(0).to(self.device)
        feat_mask = ref_a_mask.unsqueeze(0).to(self.device)

        scale_emb = (
            self.m.config.lm_config.scale_emb if self.m.config.lm_config.use_mup else 1.0
        )

        prefill_encoder = getattr(self.m, "_feat_encoder_raw", self.m.feat_encoder)
        feat_embed = prefill_encoder(feat)
        feat_embed = self.m.enc_to_lm_proj(feat_embed)

        text_embed = self.m.base_lm.embed_tokens(text_token) * scale_emb
        combined = text_mask.unsqueeze(-1) * text_embed + feat_mask.unsqueeze(-1) * feat_embed

        enc_outputs, kv_cache_tuple = self.m.base_lm(inputs_embeds=combined, is_causal=True)
        self.m.base_lm.kv_cache.fill_caches(kv_cache_tuple)

        enc_outputs = (
            self.m.fsq_layer(enc_outputs) * feat_mask.unsqueeze(-1)
            + enc_outputs * text_mask.unsqueeze(-1)
        )
        self.lm_hidden = enc_outputs[:, -1, :]

        residual_inputs = self.m.fusion_concat_proj(
            torch.cat((enc_outputs, feat_mask.unsqueeze(-1) * feat_embed), dim=-1)
        )
        residual_outputs, residual_kv_cache_tuple = self.m.residual_lm(
            inputs_embeds=residual_inputs, is_causal=True
        )
        self.m.residual_lm.kv_cache.fill_caches(residual_kv_cache_tuple)
        self.residual_hidden = residual_outputs[:, -1, :]

        # LocDiT cond starts from the last reference patch latent.
        self.prefix_feat_cond = feat[:, -1, :, :]  # [1, p, d]

        # Seed VAE decoder with last few ref patches for smooth start.
        context_len = min(self.streaming_prefix_len - 1, feat.shape[1])
        if context_len > 0:
            tail = feat[:, -context_len:, :, :]  # [1, ctx, p, d]
            for k in range(context_len):
                self.pred_feat_seq.append(tail[:, k : k + 1, :, :])

    # ------------------------------------------------------------------ #
    # Text feed
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def _push_token(self, token_id: int, is_text: bool = True) -> None:
        """Extend KV cache by one token (text or audio_start_token)."""
        scale_emb = (
            self.m.config.lm_config.scale_emb if self.m.config.lm_config.use_mup else 1.0
        )
        tok = torch.tensor([[token_id]], device=self.device, dtype=torch.long)
        text_embed = self.m.base_lm.embed_tokens(tok) * scale_emb  # [1, 1, h]

        # base_lm forward_step works on [batch, hidden]; squeeze the seq dim.
        pos = torch.tensor([self.m.base_lm.kv_cache.step()], device=self.device)
        lm_h = self.m.base_lm.forward_step(text_embed[:, 0, :], pos).clone()
        self.lm_hidden = lm_h  # text positions are NOT fsq-quantized (mask=0 for feat)

        # For text positions feat_mask=0, so residual input fusion uses zeros for feat side.
        zero_feat = torch.zeros_like(text_embed[:, 0, :])
        residual_input = self.m.fusion_concat_proj(torch.cat((lm_h, zero_feat), dim=-1))
        pos_r = torch.tensor([self.m.residual_lm.kv_cache.step()], device=self.device)
        self.residual_hidden = self.m.residual_lm.forward_step(residual_input, pos_r).clone()

    @torch.inference_mode()
    def feed_text(self, text: str) -> None:
        """Tokenize text and extend the KV caches by its length.

        May be called multiple times. Between calls you may also call
        :meth:`flush_audio` to emit some audio for the text fed so far.
        """
        if self._closed:
            raise RuntimeError("StreamingInputSession is closed")
        text = (text or "").strip()
        if not text:
            return
        token_ids = self.m.text_tokenizer(text)
        for tok in token_ids:
            self._push_token(int(tok), is_text=True)

    # ------------------------------------------------------------------ #
    # Audio generation
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def flush_audio(
        self,
        max_patches: int = 200,
        min_patches: int = 0,
    ) -> Generator[torch.Tensor, None, None]:
        """Generate up to ``max_patches`` audio patches and yield waveform chunks.

        Each yielded tensor is a float32 waveform on CPU at ``sample_rate``.
        Iteration ends either when ``max_patches`` is reached or the model's
        stop predictor fires (after ``min_patches``).
        """
        if self._closed:
            raise RuntimeError("StreamingInputSession is closed")
        if self.lm_hidden is None:
            raise RuntimeError("Session not initialized — pass a reference or feed text first")

        # The first patch after pure-text context needs a <SOA> token push so
        # the model transitions into "audio generation" mode the way it was
        # trained on (basic TTS layout).
        if not self._audio_started:
            self._push_token(self.m.audio_start_token, is_text=True)
            self._audio_started = True

        patch_size = self.m.patch_size

        for i in range(max_patches):
            dit_h1 = self.m.lm_to_dit_proj(self.lm_hidden)
            dit_h2 = self.m.res_to_dit_proj(self.residual_hidden)
            dit_h = torch.cat((dit_h1, dit_h2), dim=-1)

            pred_feat = self.m.feat_decoder(
                mu=dit_h,
                patch_size=patch_size,
                cond=self.prefix_feat_cond.transpose(1, 2).contiguous(),
                n_timesteps=self.inference_timesteps,
                cfg_value=self.cfg_value,
            ).transpose(1, 2)  # [1, p, d]

            curr_embed = self.m.feat_encoder(pred_feat.unsqueeze(1))  # [1, 1, h]
            curr_embed = self.m.enc_to_lm_proj(curr_embed)

            self.pred_feat_seq.append(pred_feat.unsqueeze(1))
            self.prefix_feat_cond = pred_feat

            feat_pred = rearrange(
                pred_feat.unsqueeze(1), "b t p d -> b d (t p)", p=patch_size
            )
            decode_audio = self._vae_dec.decode_chunk(feat_pred.to(torch.float32))
            yield decode_audio.squeeze(1).cpu()

            # Stop predictor
            stop_logits = self.m.stop_proj(self.lm_hidden)
            stop_flag = (
                self.m.stop_head(self.m.stop_actn(stop_logits)).argmax(dim=-1)[0].item()
            )
            if i >= min_patches and stop_flag == 1:
                break

            # Advance LM and RALM with the audio embed we just produced.
            pos = torch.tensor([self.m.base_lm.kv_cache.step()], device=self.device)
            lm_h = self.m.base_lm.forward_step(curr_embed[:, 0, :], pos).clone()
            self.lm_hidden = self.m.fsq_layer(lm_h)
            residual_input = self.m.fusion_concat_proj(
                torch.cat((self.lm_hidden, curr_embed[:, 0, :]), dim=-1)
            )
            pos_r = torch.tensor([self.m.residual_lm.kv_cache.step()], device=self.device)
            self.residual_hidden = self.m.residual_lm.forward_step(
                residual_input, pos_r
            ).clone()

            # Trim feat seq to streaming_prefix_len for memory.
            if len(self.pred_feat_seq) > self.streaming_prefix_len:
                self.pred_feat_seq = self.pred_feat_seq[-self.streaming_prefix_len :]

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._vae_ctx.__exit__(None, None, None)
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
