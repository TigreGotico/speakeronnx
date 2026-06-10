"""Core speaker embedding engine — pure onnxruntime + numpy.

No torch, no librosa, no soundfile at runtime.
Audio loading: stdlib ``wave`` module (PCM WAV) + numpy resampling.
Feature frontend: 80-dim log-Mel filterbank (Fbank) with cepstral mean
normalisation (CMN) — matches the WeSpeaker training pipeline exactly.
"""

from __future__ import annotations

import math
import wave
import struct
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# Public data model
# ---------------------------------------------------------------------------

@dataclass
class ModelEntry:
    """Registry entry describing one speaker-embedding ONNX model."""
    alias: str
    hf_repo: str
    hf_file: str
    license: str
    embed_dim: int
    sample_rate: int
    frontend: str          # "fbank80" (only supported value for now)
    num_mel_bins: int = 80
    frame_length_ms: float = 25.0
    frame_shift_ms: float = 10.0
    input_layout: str = "BTF"       # "BTF" = [B, T, F]; "BFT" = [B, F, T]
    output_index: int = 0            # which output contains the embedding
    extra_feeds: Optional[Dict[str, str]] = None  # extra ONNX feed entries, "T" = frame count
    description: str = ""

    def download(self) -> str:
        """Download the ONNX file via huggingface_hub into the shared HF cache."""
        return hf_hub_download(repo_id=self.hf_repo, filename=self.hf_file)


MODEL_REGISTRY: Dict[str, ModelEntry] = {
    "wespeaker-resnet34": ModelEntry(
        alias="wespeaker-resnet34",
        hf_repo="Wespeaker/wespeaker-voxceleb-resnet34-LM",
        hf_file="voxceleb_resnet34_LM.onnx",
        license="cc-by-4.0",
        embed_dim=256,
        sample_rate=16000,
        frontend="fbank80",
        description=(
            "WeSpeaker ResNet34 r-vector, large-margin finetuned on "
            "VoxCeleb2 Dev (5994 speakers). 6.63M params. Recommended "
            "default for English/multilingual speaker verification."
        ),
    ),
    "wespeaker-ecapa512": ModelEntry(
        alias="wespeaker-ecapa512",
        hf_repo="Wespeaker/wespeaker-ecapa-tdnn512-LM",
        hf_file="voxceleb_ECAPA512_LM.onnx",
        license="cc-by-4.0",
        embed_dim=192,
        sample_rate=16000,
        frontend="fbank80",
        description=(
            "WeSpeaker ECAPA-TDNN-512 x-vector, large-margin finetuned on "
            "VoxCeleb2 Dev (5994 speakers). 6.19M params. Good alternative "
            "to the ResNet34 model."
        ),
    ),
    # ------------------------------------------------------------------
    # Issue #4 — WeSpeaker ResNet293
    # ------------------------------------------------------------------
    "wespeaker-resnet293": ModelEntry(
        alias="wespeaker-resnet293",
        hf_repo="Wespeaker/wespeaker-voxceleb-resnet293-LM",
        hf_file="voxceleb_resnet293_LM.onnx",
        license="cc-by-4.0",
        embed_dim=256,
        sample_rate=16000,
        frontend="fbank80",
        description=(
            "WeSpeaker ResNet293 r-vector, large-margin finetuned on "
            "VoxCeleb2 Dev (5994 speakers). 28.62M params. The largest "
            "WeSpeaker model — best accuracy at higher compute cost."
        ),
    ),
    # ------------------------------------------------------------------
    # Issue #2 — CAM++ (WeSpeaker variant, English)
    # ------------------------------------------------------------------
    "campplus": ModelEntry(
        alias="campplus",
        hf_repo="csukuangfj/speaker-embedding-models",
        hf_file="wespeaker_en_voxceleb_CAM++_LM.onnx",
        license="cc-by-4.0",
        embed_dim=512,
        sample_rate=16000,
        frontend="fbank80",
        description=(
            "WeSpeaker CAM++ (large-margin finetuned) on VoxCeleb2 Dev. "
            "D-TDNN backbone with multi-granularity pooling. 512-dim "
            "embedding. English, VoxCeleb trained."
        ),
    ),
    # ------------------------------------------------------------------
    # Issue #2 — CAM++ (3D-Speaker multilingual)
    # ------------------------------------------------------------------
    "campplus-zh-en": ModelEntry(
        alias="campplus-zh-en",
        hf_repo="csukuangfj/speaker-embedding-models",
        hf_file="3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
        license="apache-2.0",
        embed_dim=192,
        sample_rate=16000,
        frontend="fbank80",
        description=(
            "3D-Speaker CAM++ advanced multilingual model (zh+en). "
            "D-TDNN backbone, 192-dim embedding, Apache-2.0. "
            "Good for Mandarin/English mixed verification."
        ),
    ),
    # ------------------------------------------------------------------
    # Issue #5 — ERes2Net (English, VoxCeleb)
    # ------------------------------------------------------------------
    "eres2net": ModelEntry(
        alias="eres2net",
        hf_repo="csukuangfj/speaker-embedding-models",
        hf_file="3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx",
        license="apache-2.0",
        embed_dim=192,
        sample_rate=16000,
        frontend="fbank80",
        description=(
            "3D-Speaker ERes2Net trained on VoxCeleb. 192-dim embedding. "
            "Apache-2.0. Lightweight English speaker verification."
        ),
    ),
    # ------------------------------------------------------------------
    # Issue #3 — TitaNet small (NeMo)
    # ------------------------------------------------------------------
    "titanet-small": ModelEntry(
        alias="titanet-small",
        hf_repo="csukuangfj/speaker-embedding-models",
        hf_file="nemo_en_titanet_small.onnx",
        license="cc-by-4.0",
        embed_dim=192,
        sample_rate=16000,
        frontend="fbank80",
        input_layout="BFT",
        output_index=1,
        extra_feeds={"length": "T"},
        description=(
            "NVIDIA NeMo TitaNet-small speaker embedding model. "
            "192-dim embedding. Requires transposed input [B,80,T] "
            "and frame-count length tensor. ~40 MB."
        ),
    ),
    # ------------------------------------------------------------------
    # Issue #3 — TitaNet large (NeMo)
    # ------------------------------------------------------------------
    "titanet-large": ModelEntry(
        alias="titanet-large",
        hf_repo="csukuangfj/speaker-embedding-models",
        hf_file="nemo_en_titanet_large.onnx",
        license="cc-by-4.0",
        embed_dim=192,
        sample_rate=16000,
        frontend="fbank80",
        input_layout="BFT",
        output_index=1,
        extra_feeds={"length": "T"},
        description=(
            "NVIDIA NeMo TitaNet-large speaker embedding model. "
            "192-dim embedding. Larger capacity than titanet-small. "
            "~101 MB."
        ),
    ),
    # ------------------------------------------------------------------
    # Issue #6 — ReDimNet b2 (compact high-accuracy)
    # ------------------------------------------------------------------
    "redimnet-b2": ModelEntry(
        alias="redimnet-b2",
        hf_repo="OpenVoiceOS/redimnet-b2-vox2-onnx",
        hf_file="redimnet_b2_vox2.onnx",
        license="apache-2.0",
        embed_dim=192,
        sample_rate=16000,
        frontend="raw",
        description=(
            "ReDimNet b2 (Reshape Dimensions Network) from Interspeech 2024. "
            "Compact high-accuracy model (1.8M params). Accepts raw 16kHz "
            "audio waveform with internal MelSpectrogram frontend (72 mel "
            "bins, f_max=7600). 192-dim embedding. Apache-2.0."
        ),
    ),
}

DEFAULT_MODEL = "wespeaker-resnet34"

# ---------------------------------------------------------------------------
# Audio loading — stdlib wave only, no librosa/soundfile
# ---------------------------------------------------------------------------

def _load_wav(path: str) -> Tuple[np.ndarray, int]:
    """Load a WAV file into a float32 numpy array normalised to [-1, 1].

    Supports mono and stereo PCM WAV (8/16/24/32-bit integer, 32-bit float).
    Returns (samples, sample_rate). Stereo is down-mixed to mono.
    """
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()  # bytes per sample per channel
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    total_samples = n_frames * n_channels

    if sample_width == 1:
        # unsigned 8-bit
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 3:
        # 24-bit — unpack manually
        vals = []
        for i in range(0, len(raw), 3):
            b0, b1, b2 = raw[i], raw[i + 1], raw[i + 2]
            v = (b2 << 16) | (b1 << 8) | b0
            if v & 0x800000:
                v -= 0x1000000
            vals.append(v)
        data = np.array(vals, dtype=np.float32) / 8388608.0
    elif sample_width == 4:
        # Could be int32 or float32; try float first heuristic
        try:
            data = np.frombuffer(raw, dtype=np.float32)
            if np.max(np.abs(data)) > 2.0:
                raise ValueError
        except (ValueError, TypeError):
            data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width} bytes")

    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)

    return data.astype(np.float32), framerate


def _resample_numpy(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Linear resampling via numpy (fallback when soxr not available).

    For high-quality resampling install ``soxr`` (optional dep).
    """
    if orig_sr == target_sr:
        return audio
    try:
        import soxr
        return soxr.resample(audio, orig_sr, target_sr, quality="HQ").astype(np.float32)
    except ImportError:
        pass
    # Simple linear interpolation resampler
    ratio = target_sr / orig_sr
    n_out = int(len(audio) * ratio)
    x_old = np.linspace(0, len(audio) - 1, len(audio))
    x_new = np.linspace(0, len(audio) - 1, n_out)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def load_audio(
    source: Union[str, np.ndarray],
    target_sr: int = 16000,
) -> np.ndarray:
    """Load audio from a file path or accept a pre-loaded float32 numpy array.

    Parameters
    ----------
    source:
        Path to a WAV file, or a float32 numpy array already at *target_sr*.
    target_sr:
        Desired sample rate in Hz (model's expected rate).

    Returns
    -------
    np.ndarray
        1-D float32 array, sample rate == target_sr.
    """
    if isinstance(source, np.ndarray):
        return source.astype(np.float32)
    audio, sr = _load_wav(source)
    if sr != target_sr:
        audio = _resample_numpy(audio, sr, target_sr)
    return audio


# ---------------------------------------------------------------------------
# Fbank feature frontend — matches WeSpeaker training pipeline
# ---------------------------------------------------------------------------

def _hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(
    num_mel_bins: int,
    n_fft: int,
    sample_rate: int,
    low_freq: float = 20.0,
    high_freq: Optional[float] = None,
) -> np.ndarray:
    """Build a [num_mel_bins, n_fft//2+1] mel filterbank matrix."""
    if high_freq is None:
        high_freq = sample_rate / 2.0
    mel_low = _hz_to_mel(low_freq)
    mel_high = _hz_to_mel(high_freq)
    mel_points = np.linspace(mel_low, mel_high, num_mel_bins + 2)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    fbank = np.zeros((num_mel_bins, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, num_mel_bins + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]
        for k in range(f_m_minus, f_m):
            if f_m - f_m_minus > 0:
                fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus - f_m > 0:
                fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
    return fbank


# Cache filterbanks to avoid recomputation
_FBANK_CACHE: Dict[Tuple, np.ndarray] = {}


def compute_fbank(
    audio: np.ndarray,
    sample_rate: int = 16000,
    num_mel_bins: int = 80,
    frame_length_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
    dither: float = 0.0,
    apply_cmn: bool = True,
) -> np.ndarray:
    """Compute 80-dim log-Mel filterbank features with optional CMN.

    Parameters
    ----------
    audio:
        1-D float32 waveform at *sample_rate*.
    sample_rate:
        Sample rate in Hz.
    num_mel_bins:
        Number of mel filterbank channels (80 for WeSpeaker models).
    frame_length_ms:
        Frame length in milliseconds.
    frame_shift_ms:
        Frame shift / hop size in milliseconds.
    dither:
        Amount of dithering noise (0.0 = disabled, matches WeSpeaker eval).
    apply_cmn:
        Apply cepstral mean normalisation per utterance (subtracts mean
        across time axis). WeSpeaker uses per-utterance CMN at inference.

    Returns
    -------
    np.ndarray
        Shape ``(T, num_mel_bins)`` float32 feature matrix.
    """
    frame_length = int(sample_rate * frame_length_ms / 1000)
    frame_shift = int(sample_rate * frame_shift_ms / 1000)
    n_fft = 1
    while n_fft < frame_length:
        n_fft *= 2

    # Build / retrieve filterbank
    fb_key = (num_mel_bins, n_fft, sample_rate)
    if fb_key not in _FBANK_CACHE:
        _FBANK_CACHE[fb_key] = _mel_filterbank(num_mel_bins, n_fft, sample_rate)
    fbank_mat = _FBANK_CACHE[fb_key]

    if dither > 0.0:
        audio = audio + dither * np.random.randn(len(audio)).astype(np.float32)

    # Pre-emphasis
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1]).astype(np.float32)

    # Frame the signal
    n_frames = 1 + (len(audio) - frame_length) // frame_shift
    if n_frames <= 0:
        raise ValueError(
            f"Audio too short ({len(audio)} samples) for frame_length={frame_length}"
        )

    # Build frame matrix using stride tricks
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(n_frames, frame_length),
        strides=(audio.strides[0] * frame_shift, audio.strides[0]),
    ).copy()

    # Hamming window
    window = np.hamming(frame_length).astype(np.float32)
    frames *= window

    # Power spectrum
    mag = np.abs(np.fft.rfft(frames, n=n_fft)).astype(np.float32)
    power = mag ** 2  # (T, n_fft//2+1)

    # Apply mel filterbank
    mel_power = power @ fbank_mat.T  # (T, num_mel_bins)

    # Log with floor for numerical stability
    mel_power = np.maximum(mel_power, 1e-10)
    log_mel = np.log(mel_power).astype(np.float32)

    # Per-utterance cepstral mean normalisation (subtract time-axis mean)
    if apply_cmn:
        log_mel -= log_mel.mean(axis=0, keepdims=True)

    return log_mel


# ---------------------------------------------------------------------------
# Cosine similarity helpers
# ---------------------------------------------------------------------------

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two embedding vectors, in [-1, 1]."""
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def verify(
    a: np.ndarray,
    b: np.ndarray,
    threshold: float = 0.45,
) -> Tuple[bool, float]:
    """Verify whether two embeddings belong to the same speaker.

    Parameters
    ----------
    a, b:
        Speaker embedding vectors (from :meth:`SpeakerEmbedder.embed`).
    threshold:
        Cosine similarity threshold. Scores above this are accepted as the
        same speaker. Default 0.45 is a reasonable starting point; tune on
        your own data for the desired FAR/FRR trade-off.

    Returns
    -------
    (is_same_speaker, score):
        ``is_same_speaker`` is True when ``score >= threshold``.
    """
    score = cosine(a, b)
    return score >= threshold, score


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SpeakerEmbedder:
    """Extract speaker embeddings using an ONNX model.

    Parameters
    ----------
    model:
        Alias from ``MODEL_REGISTRY`` (e.g. ``"wespeaker-resnet34"``) or
        an absolute path to a ``.onnx`` file.
    providers:
        ONNX Runtime execution providers list.  Defaults to CPU.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        providers: Optional[List[str]] = None,
    ):
        if providers is None:
            providers = ["CPUExecutionProvider"]

        if model in MODEL_REGISTRY:
            self._entry = MODEL_REGISTRY[model]
            onnx_path = self._entry.download()
        else:
            # Treat as a filesystem path
            if not os.path.isfile(model):
                raise FileNotFoundError(f"ONNX model not found: {model!r}")
            self._entry = None
            onnx_path = model

        self._session = ort.InferenceSession(onnx_path, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        output_idx = self._entry.output_index if self._entry else 0
        self._output_name = self._session.get_outputs()[output_idx].name

    @property
    def entry(self) -> Optional[ModelEntry]:
        """The registry entry for this model, or None if loaded from a path."""
        return self._entry

    @property
    def sample_rate(self) -> int:
        """Expected audio sample rate in Hz."""
        if self._entry:
            return self._entry.sample_rate
        return 16000

    @property
    def embed_dim(self) -> Optional[int]:
        """Embedding dimension, or None if loaded from a custom path."""
        if self._entry:
            return self._entry.embed_dim
        return None

    def embed(self, source: Union[str, np.ndarray]) -> np.ndarray:
        """Extract a speaker embedding vector.

        Parameters
        ----------
        source:
            Path to a WAV file, or a float32 numpy array at the model's
            expected sample rate (:attr:`sample_rate`).

        Returns
        -------
        np.ndarray
            L2-normalised 1-D float32 embedding vector.
        """
        audio = load_audio(source, target_sr=self.sample_rate)

        entry = self._entry
        if entry is None or entry.frontend == "fbank80":
            num_mel_bins = entry.num_mel_bins if entry else 80
            frame_length_ms = entry.frame_length_ms if entry else 25.0
            frame_shift_ms = entry.frame_shift_ms if entry else 10.0
            feats = compute_fbank(
                audio,
                sample_rate=self.sample_rate,
                num_mel_bins=num_mel_bins,
                frame_length_ms=frame_length_ms,
                frame_shift_ms=frame_shift_ms,
                dither=0.0,
                apply_cmn=True,
            )
        elif entry.frontend == "raw":
            feats = audio  # pass raw waveform directly
        else:
            raise ValueError(f"Unknown frontend: {entry.frontend!r}")

        # Handle input shaping / transposition
        if entry and entry.frontend == "raw":
            feats_in = feats[np.newaxis, np.newaxis, :].astype(np.float32)
        elif entry and entry.input_layout == "BFT":
            feats_in = feats.T[np.newaxis, :, :].astype(np.float32)
        else:
            feats_in = feats[np.newaxis, :, :].astype(np.float32)

        # Build feed dict
        feed_dict = {self._input_name: feats_in}
        if entry and entry.extra_feeds:
            for name, value in entry.extra_feeds.items():
                if value == "T":
                    feed_dict[name] = np.array([feats.shape[0]], dtype=np.int64)
                else:
                    raise ValueError(f"Unknown extra_feed value: {value!r}")

        out = self._session.run([self._output_name], feed_dict)
        emb = out[0].ravel().astype(np.float32)

        # L2 normalise
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb /= norm
        return emb

    def cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two embeddings (instance method shortcut)."""
        return cosine(a, b)

    def verify(
        self,
        a: np.ndarray,
        b: np.ndarray,
        threshold: float = 0.45,
    ) -> Tuple[bool, float]:
        """Verify whether two embeddings are from the same speaker."""
        return verify(a, b, threshold=threshold)
