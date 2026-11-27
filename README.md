# speakeronnx

speakeronnx is a speaker embedding library built on `onnxruntime`. It does not need
torch at runtime.

The library extracts speaker embeddings from audio, computes cosine similarity between
them, and verifies speaker identity. It downloads ONNX models from HuggingFace
automatically.

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
from urllib.request import urlretrieve
from speakeronnx import SpeakerEmbedder, verify
from speakeronnx.embedder import load_audio
clips = "https://raw.githubusercontent.com/TigreGotico/voiceclonnx/0078dec/demo/"
aria, sonia = (load_audio(urlretrieve(f"{clips}reference_{name}.wav")[0]) for name in ("aria", "sonia"))
embedder = SpeakerEmbedder("wespeaker-resnet34")
first, second, other = embedder.embed(aria[:56000]), embedder.embed(aria[56000:]), embedder.embed(sonia)
print(verify(first, second))  # (True, 0.7747...)  same speaker
print(verify(first, other))   # (False, 0.1850...) different speaker
```

The example cuts one 16 kHz clip of a speaker into two parts at 3.5 s, and compares
both parts with a clip of a second speaker. `verify` accepts a pair when the cosine
similarity is 0.45 or more.

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

The library registers 13 models in `MODEL_REGISTRY`. Each one downloads on first use:

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
| `campplus-zh-en-v2` | 192 | fbank80 | apache-2.0 |
| `eres2net-v2` | 192 | fbank80 | apache-2.0 |
| `titanet-small-v2` | 192 | fbank80 | cc-by-4.0 |
| `titanet-large-v2` | 192 | fbank80 | cc-by-4.0 |

### Aliases keep their embeddings

An alias never changes the embeddings it gives. A stored profile, such as an enrolled speaker, stays valid for as long as you use the same alias.

The `-v2` aliases use the same ONNX file with the window and normalisation of the training recipe. Use them for a new enrolment:

| For a new enrolment, choose | Instead of | Frontend of the `-v2` alias |
|---|---|---|
| `campplus-zh-en-v2` | `campplus-zh-en` | Povey window, `cmn` |
| `eres2net-v2` | `eres2net` | Povey window, `cmn` |
| `titanet-small-v2` | `titanet-small` | Hann window, `per_feature` |
| `titanet-large-v2` | `titanet-large` | Hann window, `per_feature` |

The aliases without `-v2` keep the Hamming window and `cmn`, so profiles enrolled with them still match their speakers. Embeddings from an alias and its `-v2` alias are not comparable. To move a profile to a `-v2` alias, enrol the speaker again. `ModelEntry.frontend_revision` is 1 for the old frontend and 2 for the `-v2` frontend. A store can record it next to each profile.

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

Each registered model has an HF source, license, and embedding dimension:

| Alias | HF repo | License | Embed dim | Description |
|---|---|---|---|---|
| `wespeaker-resnet34` | [Wespeaker/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM) | cc-by-4.0 | 256 | ResNet34 r-vector, VoxCeleb2 Dev - **recommended default** |
| `wespeaker-ecapa512` | [Wespeaker/wespeaker-ecapa-tdnn512-LM](https://huggingface.co/Wespeaker/wespeaker-ecapa-tdnn512-LM) | cc-by-4.0 | 192 | ECAPA-TDNN-512 x-vector, VoxCeleb2 Dev |
| `wespeaker-resnet293` | [Wespeaker/wespeaker-voxceleb-resnet293-LM](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet293-LM) | cc-by-4.0 | 256 | ResNet293 r-vector - highest accuracy, 28M params |
| `campplus` | [csukuangfj/speaker-embedding-models](https://huggingface.co/csukuangfj/speaker-embedding-models) | cc-by-4.0 | 512 | CAM++ (D-TDNN backbone), VoxCeleb2 Dev |
| `campplus-zh-en` | [csukuangfj/speaker-embedding-models](https://huggingface.co/csukuangfj/speaker-embedding-models) | apache-2.0 | 192 | 3D-Speaker CAM++ multilingual (zh+en) |
| `eres2net` | [csukuangfj/speaker-embedding-models](https://huggingface.co/csukuangfj/speaker-embedding-models) | apache-2.0 | 192 | ERes2Net, VoxCeleb |
| `titanet-small` | [csukuangfj/speaker-embedding-models](https://huggingface.co/csukuangfj/speaker-embedding-models) | cc-by-4.0 | 192 | NVIDIA NeMo TitaNet-small (~40 MB) |
| `titanet-large` | [csukuangfj/speaker-embedding-models](https://huggingface.co/csukuangfj/speaker-embedding-models) | cc-by-4.0 | 192 | NVIDIA NeMo TitaNet-large (~101 MB) |
| `redimnet-b2` | [OpenVoiceOS/redimnet-b2-vox2-onnx](https://huggingface.co/OpenVoiceOS/redimnet-b2-vox2-onnx) | apache-2.0 | 192 | ReDimNet b2 (1.8M params), raw audio input |
| `campplus-zh-en-v2`, `eres2net-v2`, `titanet-small-v2`, `titanet-large-v2` | as the alias without `-v2` | as the alias without `-v2` | 192 | Same ONNX file, frontend of the training recipe |

Run `speakeronnx list` to print descriptions and metadata for all registered models.

## Feature frontend

- **fbank80** models use an 80-dim log-Mel filterbank with the window and
  normalisation that the model entry names, implemented in pure numpy. See [`docs/frontend.md`](docs/frontend.md).
- **raw** models (redimnet-b2) pass the raw 16 kHz waveform directly
  to ONNX. The model has an internal MelSpectrogram.

## Audio requirements

- Mono PCM WAV, any bit depth (8/16/24/32-bit int, 32-bit float)
- Any sample rate (the library resamples internally to 16 kHz)
- The library downmixes stereo files to mono
- Minimum length ~1 second
- Recommended enrollment length 5-30 seconds per speaker

## Dependencies

- `onnxruntime`
- `numpy`
- `huggingface_hub`
- `soxr` (optional, for high-quality resampling)

## Related projects

- **GitHub:** [TigreGotico/speakeronnx](https://github.com/TigreGotico/speakeronnx)
- **PyPI:** `pip install speakeronnx`
- [TigreGotico/speechonnxmetrics](https://github.com/TigreGotico/speechonnxmetrics) uses speakeronnx for its speaker-similarity metric.
