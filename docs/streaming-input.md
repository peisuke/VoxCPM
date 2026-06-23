# Streaming Text Input for VoxCPM2 — Design Doc

**Status:** Draft
**Author:** peisuke (with Claude)
**Date:** 2026-06-21
**Branch:** `streaming-input`

---

## 1. ゴール

VoxCPM2 に **input-streaming TTS** 機能を追加する。LLM が出力中(=テキストがまだ全部揃っていない)状態でも、到着したテキスト断片を順次食わせて音声合成を継続できるようにする。

主な狙い:
- 対話 / AITuber / 配信用途で TTFA(Time-To-First-Audio)を 200-500ms 程度まで下げる
- VoxCPM2 の強み(高品質 voice cloning、48kHz、30言語、自然な抑揚)を保持
- Voice Design および参考音声 cloning の機能はそのまま動く(=既存ユーザの体験を壊さない)

非ゴール:
- 既存 VoxCPM2 モデル本体の音響表現や音質を変えること
- AudioVAE V2 や LocDiT のアーキテクチャに手を入れること
- フル多話者音声(VibeVoice ベース版が担う領域)

---

## 2. 背景と先行研究

### 2.1 VoxCPM2 の現状(arXiv:2606.06928)

階層的 diffusion-autoregressive モデル(2B):

```
[Text] ─► TSLM(MiniCPM-4-1B, causal RoPE, 28L H=2048)
              │
              ▼ FSQ bottleneck(dim=512, 9 levels)
              │
         hFSQ_i, hresidual_i ◄── RALM(causal NoPE, 8L H=2048)
              │
              ▼
         LocDiT(12L H=1024, diffusion head)
              │
              ▼
         Continuous latent patch z_i (4 frames @ 25Hz = 1 patch @ 6.25Hz = 160ms audio)
              │
              ▼
         AudioVAE V2 decoder ─► 48kHz waveform
```

- **出力ストリーミング**: 論文 §3.7 で「causal TSLM/RALM + patch-local LocDiT が chunk-based streaming に自然対応」と明言。nano-vllm-voxcpm で RTF≈0.13(RTX 4090)を達成済み
- **入力ストリーミング**: ライブラリ API(`generate(text=...)` / `generate_streaming(text=...)`)はテキスト全文を入力前提。途中追加食わせは未対応

### 2.2 VibeVoice-Realtime の手法(ICLR 2026)

Microsoft VibeVoice-Realtime-0.5B (`microsoft/VibeVoice-Realtime-0.5B`) は次の特徴を持つ:

- σ-VAE acoustic tokenizer(連続 latent @ 7.5Hz)、Semantic tokenizer を撤去した単純化版
- **interleaved windowed design**: テキストチャンクを逐次エンコードしつつ、過去文脈から拡散ベース acoustic latent を並列生成
- TTFA 約 200ms
- ただし voice prompt が事前計算済み `.pt` 埋め込みに限定されており、ユーザの参考音声をそのまま声色として使えない

### 2.3 アーキテクチャ比較

|              | VoxCPM2                       | VibeVoice-Realtime           |
|--------------|-------------------------------|------------------------------|
| 音表現       | 連続 latent(AudioVAE V2)    | 連続 latent(σ-VAE)          |
| パッチレート | 6.25Hz(160ms/patch)         | 7.5Hz                        |
| 言語モデル   | TSLM(causal LLM) + RALM     | 単一 causal LLM              |
| 音生成       | LocDiT 拡散ヘッド             | 拡散ヘッド                   |
| 入力スト対応 | 未学習(本ドキュメントの対象) | 訓練済                       |
| 声クローン   | 参考音声 + REF_START/END     | 専用 `.pt` 埋め込み(非公開)|
| ライセンス   | Apache 2.0                    | MIT                          |

両者の音表現は思想的に近い(連続 latent + 拡散ヘッド)。本質的な差は **訓練データのテキスト/音声の並べ方** に集約される。

---

## 3. アプローチ

VibeVoice のコードは使わず、その **設計知見(windowed interleaved 訓練データ)** だけを取り入れて、VoxCPM2 の重みとアーキテクチャを温存したまま fine-tune する。

### 3.1 温存する部品

以下は変更しない:

- AudioVAE V2(encoder/decoder 両方)
- LocEnc / LocDiT
- RALM(基本は凍結、必要なら最低限の調整)
- FSQ bottleneck
- 参考音声 pathway(REF_START / REF_END)
- vocab / tokenizer

### 3.2 変更/追加する部品

- **`AudioFeatureProcessingPacker` に `process_tts_data_streaming` を新設**
  - 既存の `process_tts_data` は「テキスト全文 → SOA → 音声パッチ列 → EOA」
  - 新タスクは interleaved: `text_chunk_0 → audio_chunks_0 → text_chunk_1 → audio_chunks_1 → ...`
  - チャンク境界マーカは既存 token id(101, 102, 103, 104)で構築可能
- **TSLM の attention mask**
  - 引き続き causal でOK(過去全部に attention、未来 text にはアクセス不可)
- **訓練ループ**
  - `task_id_map` に `"tts_streaming": 2` を追加
  - 既存の `train_voxcpm_finetune.py` をデータセット振り分けに対応させる
  - LoRA on TSLM(`LoRAConfigV2`)から実験開始
- **推論 API**
  - `core.py` に `generate_streaming_input(text_iter, ...)` を新設
  - async generator もしくは WebSocket で text chunk を受信
  - 受信のたびに TSLM の KV cache を伸ばし、LocDiT を patch 単位で動かして AudioVAE V2 から WAV を逐次出力

### 3.3 訓練データのフォーマット

#### 入力(JSONL manifest 形式)

各サンプルが既存 manifest と同じ列(audio path, text)に加えて、`chunks: List[(text_chunk, audio_start_sec, audio_end_sec)]` を持つ:

```json
{"audio": "/data/jsut/voice_001.wav",
 "text": "今日はいい天気ですね。お出かけ日和です。",
 "chunks": [
   {"text": "今日はいい", "start": 0.00, "end": 0.55},
   {"text": "天気ですね。", "start": 0.55, "end": 1.20},
   {"text": "お出かけ", "start": 1.30, "end": 1.85},
   {"text": "日和です。", "start": 1.85, "end": 2.60}
 ]}
```

#### Packer 出力(モデル入力シーケンス)

```
[ref_audio_patches] (任意, voice cloning用)
[<TEXT_CHUNK> text_token_0_0 ... text_token_0_n0]
[<SOA> audio_patch_0_0 ... audio_patch_0_m0 <EOA>]
[<TEXT_CHUNK> text_token_1_0 ... text_token_1_n1]
[<SOA> audio_patch_1_0 ... audio_patch_1_m1 <EOA>]
...
[<SOA> audio_patch_K_0 ... audio_patch_K_mK <EOA>]
[<TEXT_END>]
```

- `<TEXT_CHUNK>` は新規 special token として追加(既存 101-104 と衝突しない id を採番)
- audio チャンクの長さ `m_k` はその text chunk が読み終わるべき時間範囲を AudioVAE のパッチレート(6.25Hz)に換算
- 末端で `<TEXT_END>` を入れて「これ以上 text は来ない」を学習させる

### 3.4 チャンク境界決定

| 案 | 方法 | 特徴 |
|----|------|------|
| A. 単語境界 | 形態素境界(MeCab) + 一定文字数で打ち切り | 自然境界、軽量 |
| B. ASR 強制アラインメント | Whisper word-level timestamp / MFA | 高精度、計算コスト中 |
| C. ランダム | 均一区間で機械的に区切る | 単純だが品質低下リスク |

**採用案: A(MeCab 形態素)+ Whisper による境界補正(ハイブリッド)**

- まず MeCab で文字列を形態素列に分解
- N 形態素 or M 文字を超えたら境界候補(中央値: 5-15 char / chunk)
- Whisper-large-v3 の word-level timestamp で音声側の対応時刻を取得
- 句読点(「、」「。」「!」「?」)があれば優先的に境界に

### 3.5 訓練レシピ

| 項目 | 設定 |
|------|------|
| ベース | `openbmb/VoxCPM2`(2B fp16/bf16) |
| 学習対象 | TSLM の LoRA(rank=16, alpha=32, target=q/k/v/o + ffn) |
| 凍結 | LocEnc, RALM, LocDiT, AudioVAE V2 |
| データ混合 | 80% interleaved(新タスク) + 20% 既存 process_tts_data(catastrophic forgetting 抑制) |
| Loss | 既存と同一(LocDiT velocity loss + stop predictor + KL VAE) |
| Batch | 1〜4(audio長で変動) |
| LR | 1e-4(cosine, warmup 500step) |
| Iters | 10,000〜50,000 |
| GPU | RTX 5060 Ti 16GB(LoRA で十分収まる) |
| 想定時間 | 30〜80h(PoC スケール) |

### 3.6 推論レシピ

```python
async def generate_streaming_input(
    text_iter: AsyncIterator[str],   # LLM などからの逐次テキスト
    reference_wav_path: str = None,  # voice cloning 用(既存と同じ)
    cfg_value: float = 2.0,
    inference_timesteps: int = 10,
) -> AsyncIterator[np.ndarray]:    # 48kHz waveform chunk
    # 1. ref_audio_patches を pre-fill (cloning時)
    # 2. text_iter から chunk を受け取りつつ:
    #    - TSLM forward(KV cache 延長)
    #    - LocDiT を pacth 単位でループ → AudioVAE decode → yield wav
    # 3. text_iter 終了 → <TEXT_END> 投入 → 残り音声を吐き切って終了
```

実装上の要点:
- TSLM / RALM の KV cache を python-side で保持
- LocDiT は既存の patch-step ループをそのまま流用可能
- AudioVAE V2 は state-ful decoder なので独自に state 管理

---

## 4. フェーズ計画

### Phase 0 — 学習なし PoC(1〜2 日)

学習せず、推論側の KV cache 延長機構だけ作って挙動を観察:

- `generate_streaming_input()` のスケルトンを `core.py` に実装
- 入力テキストを人工的にチャンク分割し、`generate()` 相当の流れで KV cache を継ぎ足し
- 期待: 音声は出るがイントネーションが崩れる / 途切れる(モデルが未学習の使い方のため)
- 目的: 実装パスを早期に確定 + ベースラインの「何がダメか」を見える化

### Phase 1 — データ準備(1〜2 週)

- ReazonSpeech v2(35,000h 日本語、ライセンス確認)を取得 → 100h サブセットを最初に使用
- Whisper-large-v3 で word-level timestamp 取得
- MeCab で形態素分割
- `scripts/prepare_streaming_manifest.py` を新設して JSONL を出力
- データ統計: チャンク長分布、無音区間、句読点率を確認

### Phase 2 — Fine-tune & 評価(1〜2 週)

- `packers.py` に `process_tts_data_streaming` 追加
- `train_voxcpm_finetune.py` に streaming タスク受け入れを追加
- LoRA 訓練(30〜80h on RTX 5060 Ti)
- 評価指標:
  - **TTFA**: text 投入から最初の音声 chunk が出るまで
  - **WER / CER**: 出力音声を Whisper で再認識し、入力テキストとの誤り率
  - **SIM-O**: 参考音声と出力音声の話者埋め込み類似度(=ロリ声クローンが保たれているか)
  - **Subjective MOS**: 内輪で 5-10 サンプル比較

### Phase 3 — 本格運用(必要なら)

- 1000h+ データで full fine-tune
- WebSocket TTS サーバ実装(`demo/streaming_server.py`)
- AITuberKit との接続(custom TTS endpoint)

---

## 5. 想定リスクと緩和策

| リスク | 影響度 | 緩和策 |
|--------|--------|--------|
| LoRA だけでは streaming 学習が収束しない | 中 | TSLM フル fine-tune に格上げ |
| 既存の voice cloning 機能が劣化(catastrophic forgetting) | 中 | データ混合 80/20 + 評価で都度確認 |
| 6.25Hz パッチ境界が text chunk と揃わない | 低 | パッチを最小単位、text chunk は ≥1 patch ぶんの音声に対応する制約 |
| ReazonSpeech のライセンスが商用不可 | 中 | 個人研究目的のみで使用、商用化時は別データソース(JVS, Common Voice JP)に切替 |
| Whisper の word-timestamp 精度が低い言語混在 | 中 | 日本語に絞って実験開始、後で多言語拡張 |

---

## 6. オープン問題

- `<TEXT_CHUNK>` special token を新規追加するか、既存トークンの再利用で済ますか(=モデル側の vocab を触らずに済ますなら後者が楽)
- 推論時に text_iter が "途切れた" を判定する仕組み(タイムアウト or 明示シグナル)
- AITuberKit 等のフロントエンドへの統合インターフェイス(OpenAI 互換 `/v1/audio/speech` ストリーミング? カスタム WebSocket?)

---

## 7. 参考文献

- VoxCPM2 Technical Report (OpenBMB, 2026): arXiv:2606.06928
- VibeVoice (Microsoft, ICLR 2026): https://openreview.net/pdf?id=FihSkzyxdv
- VibeVoice-Realtime-0.5B: https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B
- ReazonSpeech v2: https://huggingface.co/datasets/reazon-research/reazonspeech
