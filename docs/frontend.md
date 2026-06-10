# Feature frontend

## Fbank80 (WeSpeaker-style)

All models except `redimnet-b2` use 80-dim log-Mel filterbank features
with per-utterance cepstral mean normalisation (CMN). This matches the
WeSpeaker training pipeline exactly.

### Processing steps

```
Raw waveform (16 kHz)
        │
        ▼
Pre-emphasis (0.97)
        │
        ▼
Framing: 25 ms windows, 10 ms shift
        │
        ▼
Hamming window
        │
        ▼
FFT (power spectrum)
        │
        ▼
Mel filterbank (80 bins, 20–8000 Hz)
        │
        ▼
Log(mel_power)
        │
        ▼
Per-utterance CMN (subtract time-axis mean)
        │
        ▼
Output: (T, 80) float32
```

### Implementation

The frontend is implemented in pure numpy in `speakeronnx.embedder.compute_fbank`.
Key design decisions:

- **No librosa/soundfile** — zero external audio processing dependencies.
- **Filterbank caching** — the mel filterbank matrix is computed once and
  cached per `(num_mel_bins, n_fft, sample_rate)` key.
- **CMN at inference** — WeSpeaker eval pipeline uses per-utterance CMN,
  not speaker-level or global statistics.
- **Dither disabled** — `dither=0.0` during inference matches WeSpeaker eval.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `num_mel_bins` | 80 | Number of mel filterbank channels |
| `frame_length_ms` | 25.0 | Frame length in milliseconds |
| `frame_shift_ms` | 10.0 | Frame shift / hop size |
| `low_freq` | 20 Hz | Lowest frequency for mel filterbank |
| `high_freq` | 8000 Hz | Highest frequency (nyquist) |

## Raw frontend

`redimnet-b2` uses a `"raw"` frontend that passes the waveform directly
to the ONNX model, which contains an internal MelSpectrogram layer.

### Processing steps

```
Raw waveform (16 kHz)
        │
        ▼
Add channel dims: [1, 1, T]
        │
        ▼
ONNX model internal MelSpectrogram
(72 mel bins, f_max=7600 Hz, n_fft=512, hop_length=128)
        │
        ▼
Output: (1, 192) embedding
```

No pre-emphasis, CMN, or external feature extraction is applied.
The model's internal frontend handles everything.

### Comparison

| Aspect | Fbank80 | Raw (ReDimNet) |
|---|---|---|
| External deps | None (pure numpy) | None (pure numpy) |
| Pre-emphasis | 0.97 | Internal to model |
| Mel bins | 80 | 72 |
| Freq range | 20–8000 Hz | 0–7600 Hz |
| CMN | Per-utterance | None |
| Frame length | 25 ms | ~32 ms (n_fft=512) |
| Frame shift | 10 ms | 8 ms (hop=128 at 16 kHz) |
| N_FFT | Adaptive (next power of 2 ≥ frame_length) | 512 (fixed in model) |
