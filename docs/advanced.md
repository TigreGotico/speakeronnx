# Advanced usage

## Custom models

Load any ONNX model from disk, not just the built-in registry:

```python
from speakeronnx import SpeakerEmbedder

embedder = SpeakerEmbedder("/path/to/custom_model.onnx")
path_emb = embedder.embed("speaker.wav")
```

When loading from a path, `embedder.entry` is `None`, and `embedder.embed_dim`
is `None`. The model must accept 16 kHz audio and return an embedding as its
first output (or use `output_index` - not configurable for custom models).

## GPU inference

ONNX Runtime supports CUDA, CoreML, and other providers:

```python
embedder = SpeakerEmbedder(
    model="wespeaker-resnet34",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
```

The providers list is passed directly to `onnxruntime.InferenceSession`.
If CUDA is unavailable, it falls back to CPU.

## Threshold tuning

The default threshold of `0.45` is a starting point. The optimal threshold depends
on your data, your language, and the false-accept / false-reject trade-off you want.

**To calibrate:**

```python
from speakeronnx import SpeakerEmbedder, cosine

emb = SpeakerEmbedder(model="wespeaker-resnet34")

same_scores = []
diff_scores = []

for same_pair in same_speaker_pairs:
    e1 = emb.embed(same_pair[0])
    e2 = emb.embed(same_pair[1])
    same_scores.append(cosine(e1, e2))

for diff_pair in diff_speaker_pairs:
    e1 = emb.embed(diff_pair[0])
    e2 = emb.embed(diff_pair[1])
    diff_scores.append(cosine(e1, e2))
```

Then choose a threshold that separates the two distributions.

## Audio requirements

- **Format:** Mono PCM WAV (any bit depth: 8/16/24/32-bit int, 32-bit float)
- **Sample rate:** Any (resampled internally to 16 kHz)
- **Minimum duration:** ~1 second (shorter audio produces too few fbank frames)
- **Recommended enrollment:** 5–30 seconds per speaker

Stereo files are down-mixed to mono automatically.

## Speech activity detection

The library does **not** perform VAD (voice activity detection). Feeding
silence or non-speech audio will produce poor embeddings. For best results,
trim silence before calling `embed()`.

## Resampling

- **`soxr` installed (recommended):** high-quality band-limited resampling
- **`soxr` not installed:** linear interpolation (adequate but lower quality)

## Determinism

Embeddings are deterministic for a given model and input file. Multiple calls
to `embedder.embed("same.wav")` return identical results.

## Environment variables

| Variable | Effect |
|---|---|
| `HF_HOME` | Path to HuggingFace cache (default `~/.cache/huggingface/hub`) |
| `OMP_NUM_THREADS` | Number of threads for ONNX Runtime CPU inference |

---
[← Feature frontend](frontend.md) · [Home](index.md)
