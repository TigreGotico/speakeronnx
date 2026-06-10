# speakeronnx

Pure-onnxruntime speaker embedding library — no torch at runtime.

Extract speaker embeddings, compute cosine similarity, and verify speaker identity
using ONNX-exported models downloaded automatically from HuggingFace.

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
from speakeronnx import SpeakerEmbedder, cosine, verify

embedder = SpeakerEmbedder(model="wespeaker-resnet34")

alice1 = embedder.embed("alice_clip1.wav")
alice2 = embedder.embed("alice_clip2.wav")
bob    = embedder.embed("bob_clip1.wav")

print(cosine(alice1, alice2))   # e.g. 0.82 — same speaker
print(cosine(alice1, bob))      # e.g. 0.21 — different speaker

ok, score = verify(alice1, alice2, threshold=0.45)
print(ok, score)  # True 0.82
```

## Models

All models are downloaded on first use into the shared HuggingFace cache (`HF_HOME`).

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
