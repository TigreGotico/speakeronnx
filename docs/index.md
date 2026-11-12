# speakeronnx

speakeronnx is a speaker embedding library built on `onnxruntime`. It does not need
torch at runtime.

The library extracts speaker embeddings from audio, computes cosine similarity between
them, and verifies speaker identity. It downloads ONNX models from HuggingFace
automatically.

## Installation

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

## Models

The library downloads each model on first use into the shared HuggingFace cache
(`HF_HOME`).

| Alias | Embed dim | Params | Frontend | License |
|---|---|---|---|---|
| `wespeaker-resnet34` | 256 | 6.6M | fbank80 | cc-by-4.0 |
| `wespeaker-ecapa512` | 192 | 6.2M | fbank80 | cc-by-4.0 |
| `wespeaker-resnet293` | 256 | 28.6M | fbank80 | cc-by-4.0 |
| `campplus` | 512 | ~6M | fbank80 | cc-by-4.0 |
| `campplus-zh-en` | 192 | ~3M | fbank80 | apache-2.0 |
| `eres2net` | 192 | ~5M | fbank80 | apache-2.0 |
| `titanet-small` | 192 | ~3M | fbank80 | cc-by-4.0 |
| `titanet-large` | 192 | ~8M | fbank80 | cc-by-4.0 |
| `redimnet-b2` | 192 | 1.8M | raw | apache-2.0 |

See [models.md](models.md) for detailed model comparison and selection guidance.

## CLI

```bash
speakeronnx list                              # list available models
speakeronnx embed clip.wav                    # extract embedding
speakeronnx verify a.wav b.wav               # same-speaker check (exit 0/1)
speakeronnx verify a.wav b.wav --threshold 0.5
speakeronnx embed clip.wav --model wespeaker-ecapa512
```

## Project links

- **GitHub:** [TigreGotico/speakeronnx](https://github.com/TigreGotico/speakeronnx)
- **HF Collection:** [OpenVoiceOS/speaker-embeddings-onnx](https://huggingface.co/collections/OpenVoiceOS/speaker-embeddings-onnx)
- **PyPI:** `pip install speakeronnx`
