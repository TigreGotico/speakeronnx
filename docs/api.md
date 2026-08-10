# API reference

## `speakeronnx` top-level

```python
from speakeronnx import SpeakerEmbedder, cosine, verify, MODEL_REGISTRY, ModelEntry
```

### `MODEL_REGISTRY`

```python
MODEL_REGISTRY: Dict[str, ModelEntry]
```

A dict that maps alias strings to `ModelEntry` dataclass instances.
All 9 built-in models are registered here.

### `ModelEntry`

```python
@dataclass
class ModelEntry:
    alias: str
    hf_repo: str
    hf_file: str
    license: str
    embed_dim: int
    sample_rate: int
    frontend: str                # "fbank80" or "raw"
    num_mel_bins: int = 80
    frame_length_ms: float = 25.0
    frame_shift_ms: float = 10.0
    input_layout: str = "BTF"    # "BTF" or "BFT"
    output_index: int = 0
    extra_feeds: Optional[Dict[str, str]] = None
    description: str = ""
```

| Field | Description |
|---|---|
| `alias` | Short name used to select the model (matches dict key) |
| `hf_repo` | HuggingFace repo ID |
| `hf_file` | ONNX filename within the repo |
| `license` | SPDX license identifier |
| `embed_dim` | Output embedding dimensionality |
| `sample_rate` | Expected audio sample rate (Hz) |
| `frontend` | Feature frontend: `"fbank80"` or `"raw"` |
| `num_mel_bins` | Fbank mel bins (fbank80 models only) |
| `frame_length_ms` | Fbank frame length (fbank80 models only) |
| `frame_shift_ms` | Fbank frame shift (fbank80 models only) |
| `input_layout` | ONNX input tensor layout: `"BTF"` or `"BFT"` |
| `output_index` | Which ONNX output is the embedding |
| `extra_feeds` | Extra ONNX feeds (e.g. `{"length": "T"}` for TitaNet) |
| `description` | Human-readable model summary |

#### `ModelEntry.download()`

```python
entry.download() -> str
```

Downloads the ONNX file via `huggingface_hub` and returns the local path.
The library caches the file in the shared HF cache (`HF_HOME`).

---

### `SpeakerEmbedder`

```python
class SpeakerEmbedder:
    def __init__(
        self,
        model: str = "wespeaker-resnet34",
        providers: Optional[List[str]] = None,
    )
```

The main class for extracting speaker embeddings.

**Parameters:**

- `model` - alias from `MODEL_REGISTRY` (e.g. `"wespeaker-resnet34"`) or
  absolute path to a custom `.onnx` file.
- `providers` - ONNX Runtime execution providers (defaults to `["CPUExecutionProvider"]`).

**Properties:**

| Property | Type | Description |
|---|---|---|
| `entry` | `Optional[ModelEntry]` | Registry entry, or `None` for custom paths |
| `sample_rate` | `int` | Model's expected sample rate (Hz) |
| `embed_dim` | `Optional[int]` | Embedding dimension, or `None` for custom paths |

#### `embed()`

```python
def embed(self, source: Union[str, np.ndarray]) -> np.ndarray
```

Extract a speaker embedding from audio.

**Parameters:**

- `source` - path to a WAV file, or a float32 numpy array at the
  model's expected sample rate.

**Returns:**

- L2-normalized 1-D float32 embedding vector.

**Processing pipeline:**

1. Load audio (WAV or numpy array) → resample to 16 kHz if needed
2. Compute features via the model's frontend (`fbank80` or `raw`)
3. Reshape/transpose input to match model's expected layout
4. Add extra ONNX feeds (TitaNet length tensor)
5. Run ONNX inference
6. L2-normalize the output embedding

#### `cosine()`

```python
def cosine(self, a: np.ndarray, b: np.ndarray) -> float
```

Instance-method shortcut for `speakeronnx.cosine(a, b)`.

#### `verify()`

```python
def verify(
    self,
    a: np.ndarray,
    b: np.ndarray,
    threshold: float = 0.45,
) -> Tuple[bool, float]
```

Instance-method shortcut for `speakeronnx.verify(a, b, threshold)`.

---

### `cosine()`

```python
def cosine(a: np.ndarray, b: np.ndarray) -> float
```

Cosine similarity between two embedding vectors, in `[-1, 1]`.

```python
from speakeronnx import cosine
similarity = cosine(embedding_a, embedding_b)
```

Returns `0.0` if either vector is zero-norm.

### `verify()`

```python
def verify(
    a: np.ndarray,
    b: np.ndarray,
    threshold: float = 0.45,
) -> Tuple[bool, float]
```

Verify whether two embeddings belong to the same speaker.

**Parameters:**

- `a, b` - speaker embedding vectors.
- `threshold` - cosine similarity threshold. Scores above this are
  accepted as the same speaker. The default `0.45` is a reasonable starting
  point. Tune it on your own data for the false-accept / false-reject
  trade-off you want.

**Returns:**

- `(is_same_speaker, score)` - `is_same_speaker` is `True` when
  `score >= threshold`.

---
[← Model guide](models.md) · [Home](index.md) · [CLI reference →](cli.md)
