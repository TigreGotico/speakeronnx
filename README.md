# speakeronnx

Pure-onnxruntime speaker embedding library — no torch at runtime.

Extract speaker embeddings, compute cosine similarity, and verify speaker identity
using ONNX-exported models downloaded automatically from HuggingFace.

**Model collection:** [OpenVoiceOS/speaker-embeddings-onnx](https://huggingface.co/collections/OpenVoiceOS/speaker-embeddings-onnx)

## Install

```bash
pip install speakeronnx
```

Optional high-quality resampling:
```bash
pip install speakeronnx soxr
```

## Quick start

```python
from speakeronnx import SpeakerEmbedder, cosine, verify

embedder = SpeakerEmbedder(model="wespeaker-resnet34")

alice1 = embedder.embed("alice_clip1.wav")
alice2 = embedder.embed("alice_clip2.wav")
bob    = embedder.embed("bob_clip1.wav")

print(cosine(alice1, alice2))   # e.g. 0.82  — same speaker
print(cosine(alice1, bob))      # e.g. 0.21  — different speaker

ok, score = verify(alice1, alice2, threshold=0.45)
print(ok, score)  # True 0.82
```

More examples in [`examples/`](examples/).

## CLI

```bash
speakeronnx list                              # list available models
speakeronnx embed clip.wav                    # extract embedding
speakeronnx verify a.wav b.wav               # same-speaker check (exit 0/1)
speakeronnx verify a.wav b.wav --threshold 0.5
speakeronnx embed clip.wav --model wespeaker-ecapa512
```

Full CLI reference in [`docs/cli.md`](docs/cli.md).

## Models

All 9 models are registered in `MODEL_REGISTRY` and downloaded on first use:

| Alias | Embed dim | Frontend | License |
|---|---|---|---|
| `wespeaker-resnet34` | 256 | fbank80 | cc-by-4.0 |
| `wespeaker-ecapa512` | 192 | fbank80 | cc-by-4.0 |
| `wespeaker-resnet293` | 256 | fbank80 | cc-by-4.0 |
| `campplus` | 512 | fbank80 | cc-by-4.0 |
| `campplus-zh-en` | 192 | fbank80 | apache-2.0 |
| `eres2net` | 192 | fbank80 | apache-2.0 |
| `titanet-small` | 192 | fbank80 | cc-by-4.0 |
| `titanet-large` | 192 | fbank80 | cc-by-4.0 |
| `redimnet-b2` | 192 | raw | apache-2.0 |

Full model comparison and selection guide in [`docs/models.md`](docs/models.md).

## Documentation

| Document | Description |
|---|---|
| [`docs/index.md`](docs/index.md) | Full getting-started guide |
| [`docs/models.md`](docs/models.md) | Model comparison, selection, frontend/layout details |
| [`docs/api.md`](docs/api.md) | Complete API reference |
| [`docs/cli.md`](docs/cli.md) | CLI usage reference |
| [`docs/frontend.md`](docs/frontend.md) | Feature frontend (fbank80 vs raw) technical details |
| [`docs/advanced.md`](docs/advanced.md) | Custom models, GPU, threshold tuning |

## Examples

| Script | Description |
|---|---|
| [`examples/basic_embedding.py`](examples/basic_embedding.py) | Extract embedding from a single WAV |
| [`examples/verify_speakers.py`](examples/verify_speakers.py) | Verify two clips, try multiple thresholds |
| [`examples/compare_models.py`](examples/compare_models.py) | Compare all models on same utterances |
| [`examples/batch_enrollment.py`](examples/batch_enrollment.py) | Enroll speakers from directories, match unknown |
| [`examples/custom_model.py`](examples/custom_model.py) | Load a custom ONNX model from disk |
| [`examples/gpu_inference.py`](examples/gpu_inference.py) | CUDA / CoreML inference |

## Tests

```bash
# Unit tests (mocked, no downloads, no network)
pytest tests/test_unit.py tests/test_audio.py tests/test_frontend.py \
      tests/test_embedder.py tests/test_cli.py tests/test_model_registry.py -v

# End-to-end tests (downloads models + generates TTS audio)
pytest tests/test_e2e.py -v -s
```

## Feature frontend

- **fbank80** models: 80-dim log-Mel filterbank with per-utterance CMN,
  implemented in pure numpy. See [`docs/frontend.md`](docs/frontend.md).
- **raw** models (redimnet-b2): raw 16 kHz waveform passed directly
  to ONNX (internal MelSpectrogram in the model).

## Audio requirements

- Mono PCM WAV, any bit depth (8/16/24/32-bit int, 32-bit float)
- Any sample rate (resampled internally to 16 kHz)
- Stereo files are downmixed to mono
- Minimum ~1 second; recommended enrollment 5–30 seconds per speaker

## Dependencies

- `onnxruntime`
- `numpy`
- `huggingface_hub`
- `soxr` (optional, for high-quality resampling)

## Project links

- **GitHub:** [TigreGotico/speakeronnx](https://github.com/TigreGotico/speakeronnx)
- **PyPI:** `pip install speakeronnx`
