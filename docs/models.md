# Model guide

All 9 models are registered in `speakeronnx.MODEL_REGISTRY` and downloaded
automatically via `huggingface_hub` on first use.

## Choosing a model

### Best accuracy (compute-heavy)

| Model | Dim | Params | Notes |
|---|---|---|---|
| `wespeaker-resnet293` | 256 | 28.6M | Highest accuracy WeSpeaker model. ResNet293 r-vector, large-margin finetuned. |
| `campplus` | 512 | ~6M | High-dim embedding. D-TDNN with multi-granularity pooling. Best when dim matters. |
| `titanet-large` | 192 | ~8M | NVIDIA NeMo large capacity. |

### Best accuracy-to-size ratio

| Model | Dim | Params | Notes |
|---|---|---|---|
| `redimnet-b2` | 192 | 1.8M | Interspeech 2024. tiny yet competitive. Raw audio input. |
| `wespeaker-resnet34` | 256 | 6.6M | Default model. Solid all-rounder. |
| `campplus-zh-en` | 192 | ~5M | Multilingual (zh+en), Apache-2.0. |

### Lightweight / fastest

| Model | Dim | Params | Notes |
|---|---|---|---|
| `wespeaker-ecapa512` | 192 | 6.2M | ECAPA-TDNN, fast inference. |
| `eres2net` | 192 | ~5M | Lightweight English-only, Apache-2.0. |
| `titanet-small` | 192 | ~3M | TitaNet small, 40 MB on disk. |

## Frontend requirements

### `fbank80` models (all except redimnet-b2)

Expect 80-dim log-Mel filterbank features with per-utterance CMN.
The library's `compute_fbank` function implements this in pure numpy.

Processing flow:
1. Load PCM WAV, resample to 16 kHz
2. Pre-emphasis (0.97)
3. Hamming window (25 ms frame, 10 ms shift)
4. Mel filterbank (80 bins, 20–8000 Hz)
5. Log power spectrum
6. Per-utterance CMN (subtract time-axis mean)

### `raw` models (redimnet-b2)

Accepts raw 16 kHz waveform. The ONNX model contains an internal
MelSpectrogram frontend (72 mel bins, f_max=7600 Hz, n_fft=512, hop=128).
No pre-emphasis or CMN applied externally.

## Input layouts

| Layout | Shape | Models |
|---|---|---|
| `BTF` | `[B, T, F]` | All WeSpeaker, CAM++, ERes2Net, ReDimNet |
| `BFT` | `[B, F, T]` | TitaNet small/large |

Layout is handled automatically by `SpeakerEmbedder.embed()`.

## Extra ONNX feeds

Some models require additional input tensors aside from the audio features:

| Model | Extra feed | Type | Value |
|---|---|---|---|
| `titanet-small` | `length` | `int64[B]` | Frame count per utterance |
| `titanet-large` | `length` | `int64[B]` | Frame count per utterance |

These are populated automatically.

## Output index

Most models have a single output (the embedding). TitaNet models have two
outputs (`logits` at index 0, `embs` at index 1). `SpeakerEmbedder` selects
the correct output automatically.

## Model sources

- **Wespeaker models**: [Wespeaker org](https://huggingface.co/Wespeaker) — cc-by-4.0
- **csukuangfj bundle**: [csukuangfj/speaker-embedding-models](https://huggingface.co/csukuangfj/speaker-embedding-models) — cc-by-4.0 / apache-2.0
- **RedimNet**: [OpenVoiceOS/redimnet-b2-vox2-onnx](https://huggingface.co/OpenVoiceOS/redimnet-b2-vox2-onnx) — apache-2.0
