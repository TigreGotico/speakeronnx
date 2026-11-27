"""An array input carries no sample rate: embed() takes it as sample_rate=."""
from dataclasses import replace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _tones(sr: int, seconds: float = 2.0) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 1300 * t)
            + 0.1 * np.sin(2 * np.pi * 3100 * t)).astype(np.float32)


@pytest.fixture
def embedder():
    """Stub session: the embedding is the time-mean of the un-normalised log-Mel
    input, so it follows the spectrum the model sees."""
    with patch("speakeronnx.embedder.hf_hub_download", return_value="/fake/model.onnx"), \
            patch("onnxruntime.InferenceSession"):
        from speakeronnx import SpeakerEmbedder
        emb = SpeakerEmbedder(model="wespeaker-resnet34")
    emb._entry = replace(emb._entry, normalisation=None)
    name = emb._input_name
    emb._session.run = MagicMock(side_effect=lambda _o, feed: [feed[name][0].mean(axis=0)])
    return emb


def test_44k_array_with_sample_rate_matches_16k(embedder):
    ref = embedder.embed(_tones(16000), sample_rate=16000)
    hi = embedder.embed(_tones(44100), sample_rate=44100)
    # Measured on this stub: 0.996 with resampling, 0.769 without.
    assert embedder.cosine(ref, hi) > 0.99


def test_array_without_sample_rate_warns(embedder):
    with pytest.warns(DeprecationWarning, match="sample_rate="):
        embedder.embed(_tones(16000))
