# speakeronnx

Pure-onnxruntime speaker embedding library — no torch at runtime.

Extract speaker embeddings, compute cosine similarity, and verify speaker identity
using ONNX-exported models downloaded automatically from HuggingFace.

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

## CLI

```bash
speakeronnx list                              # list available models
speakeronnx embed clip.wav                    # extract embedding
speakeronnx verify a.wav b.wav               # same-speaker check (exit 0/1)
speakeronnx verify a.wav b.wav --threshold 0.5
speakeronnx embed clip.wav --model wespeaker-ecapa512
```

## Models

| Alias | HF repo | License | Embed dim | Description |
|---|---|---|---|---|
| `wespeaker-resnet34` | [Wespeaker/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM) | cc-by-4.0 | 256 | ResNet34 r-vector, VoxCeleb2 Dev |
| `wespeaker-ecapa512` | [Wespeaker/wespeaker-ecapa-tdnn512-LM](https://huggingface.co/Wespeaker/wespeaker-ecapa-tdnn512-LM) | cc-by-4.0 | 192 | ECAPA-TDNN-512 x-vector, VoxCeleb2 Dev |

Models are downloaded on first use into the shared HuggingFace cache (`HF_HOME`).

## Feature frontend

WeSpeaker models expect 80-dim log-Mel filterbank (Fbank) features with per-utterance
cepstral mean normalisation (CMN). This library implements the frontend in pure numpy
with no external audio processing dependencies. Audio loading uses the stdlib `wave`
module; resampling uses linear interpolation (or `soxr` if installed).

## Audio requirements

Input WAV files should be mono PCM, any sample rate (resampled internally to 16 kHz).
Minimum recommended duration: ~1 second. For enrollment, 5–30 seconds per speaker
gives best accuracy.

## Dependencies

- `onnxruntime`
- `numpy`
- `huggingface_hub`
- `soxr` (optional, for high-quality resampling)
