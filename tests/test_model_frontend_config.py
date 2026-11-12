"""Per-model window, normalisation, threshold and revision, and the silence gate.

Sessions are mocked, so these tests download nothing.
"""
import dataclasses
import re
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from speakeronnx import MODEL_REGISTRY, SpeakerEmbedder, verify
from speakeronnx.embedder import compute_fbank
from speakeronnx.version import VERSION_MAJOR


def _embedder(alias, output=None):
    with patch("speakeronnx.embedder.hf_hub_download", return_value="/fake/model.onnx"), \
            patch("onnxruntime.InferenceSession"):
        embedder = SpeakerEmbedder(model=alias)
    vector = np.linspace(1.0, 2.0, MODEL_REGISTRY[alias].embed_dim).astype(np.float32) if output is None else output
    embedder._session.run = MagicMock(return_value=[vector])
    return embedder


def _tone(seconds=1.0, amplitude=0.3):
    t = np.arange(int(16000 * seconds)) / 16000
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


# ------------------------------------------------------------------ #
# silence
# ------------------------------------------------------------------ #
def test_two_silent_clips_fail_verify():
    # The mocked model returns one fixed vector for any input, as a real model does
    # for two identical silent inputs. Before the gate, verify accepted that pair at 1.0.
    embedder = _embedder("wespeaker-resnet34")
    silent = np.zeros(32000, dtype=np.float32)
    first, second = embedder.embed(silent), embedder.embed(silent.copy())
    assert embedder.verify(first, second) == (False, 0.0)


def test_silence_gives_the_zero_vector():
    embedder = _embedder("wespeaker-resnet34")
    emb = embedder.embed(np.zeros(32000, dtype=np.float32))
    assert emb.shape == (256,) and not np.any(emb)


def test_a_clip_at_one_16_bit_step_is_not_silence():
    embedder = _embedder("wespeaker-resnet34")
    quiet = np.zeros(32000, dtype=np.float32)
    quiet[::2] = 1 / 32768
    emb = embedder.embed(quiet)
    assert np.linalg.norm(emb) == pytest.approx(1.0)


@pytest.mark.parametrize("threshold", [0.45, 0.0, -1.0])
def test_a_zero_vector_never_verifies(threshold):
    zero, voice = np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32)
    assert verify(zero, voice, threshold=threshold) == (False, 0.0)
    assert verify(zero, zero, threshold=threshold) == (False, 0.0)


# ------------------------------------------------------------------ #
# window and normalisation
# ------------------------------------------------------------------ #
def test_povey_window_is_the_kaldi_definition():
    # Kaldi feature-window.cc: pow(0.5 - 0.5 cos(2 pi i / (N - 1)), 0.85)
    from speakeronnx.embedder import _WINDOWS
    n = np.arange(400)
    expected = (0.5 - 0.5 * np.cos(2 * np.pi * n / 399)) ** 0.85
    np.testing.assert_allclose(_WINDOWS["povey"](400), expected, atol=1e-12)


def test_per_feature_normalisation_gives_zero_mean_and_unit_std_per_bin():
    rng = np.random.default_rng(0)
    audio = rng.standard_normal(16000).astype(np.float32)
    raw = compute_fbank(audio, normalisation=None)
    feats = compute_fbank(audio, normalisation="per_feature")
    np.testing.assert_allclose(feats.mean(axis=0), 0.0, atol=1e-4)
    # A mel bin that covers no FFT bin holds the log floor in every frame and has no spread.
    spread = raw.std(axis=0) > 1e-3
    assert spread.sum() >= 70
    np.testing.assert_allclose(feats.std(axis=0, ddof=1)[spread], 1.0, atol=1e-3)


def test_unknown_normalisation_raises():
    with pytest.raises(ValueError, match="normalisation"):
        compute_fbank(_tone(), normalisation="global")


@pytest.mark.parametrize("alias,window,normalisation", [
    ("wespeaker-resnet34", "hamming", "cmn"),
    ("campplus-zh-en", "hamming", "cmn"),
    ("eres2net", "hamming", "cmn"),
    ("titanet-small", "hamming", "cmn"),
    ("titanet-large", "hamming", "cmn"),
    ("campplus-zh-en-v2", "povey", "cmn"),
    ("eres2net-v2", "povey", "cmn"),
    ("titanet-small-v2", "hann", "per_feature"),
    ("titanet-large-v2", "hann", "per_feature"),
])
def test_embed_feeds_the_features_of_the_model_entry(alias, window, normalisation):
    embedder = _embedder(alias)
    audio = _tone()
    embedder.embed(audio)
    feed = embedder._session.run.call_args[0][1][embedder._input_name][0]
    expected = compute_fbank(audio, window=window, normalisation=normalisation)
    np.testing.assert_allclose(feed.T if MODEL_REGISTRY[alias].input_layout == "BFT" else feed, expected)


@pytest.mark.parametrize("alias", ["campplus-zh-en", "eres2net", "titanet-small", "titanet-large"])
def test_a_corrected_frontend_ships_under_a_new_alias(alias):
    # Profiles stored under an alias hold its embeddings, so the alias keeps its frontend.
    old, new = MODEL_REGISTRY[alias], MODEL_REGISTRY[f"{alias}-v2"]
    assert (old.window, old.normalisation, old.frontend_revision) == ("hamming", "cmn", 1)
    assert new.frontend_revision == 2
    assert (new.hf_repo, new.hf_file, new.revision) == (old.hf_repo, old.hf_file, old.revision)


@pytest.mark.parametrize("apply_cmn,normalisation", [(True, "cmn"), (False, None)])
def test_apply_cmn_still_works_and_warns(apply_cmn, normalisation):
    audio = _tone()
    with pytest.warns(DeprecationWarning, match=rf"removed in speakeronnx {VERSION_MAJOR + 1}\.0\.0"):
        feats = compute_fbank(audio, apply_cmn=apply_cmn)
    np.testing.assert_array_equal(feats, compute_fbank(audio, normalisation=normalisation))


# ------------------------------------------------------------------ #
# threshold and revision
# ------------------------------------------------------------------ #
def test_verify_on_the_embedder_uses_the_threshold_of_the_entry():
    embedder = _embedder("wespeaker-resnet34")
    embedder._entry = dataclasses.replace(embedder._entry, threshold=0.99)
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.95, np.sqrt(1 - 0.95 ** 2)], dtype=np.float32)  # cosine 0.95
    assert embedder.verify(a, b)[0] is False
    assert embedder.verify(a, b, threshold=0.9)[0] is True


@pytest.mark.parametrize("alias", sorted(MODEL_REGISTRY))
def test_every_model_download_is_pinned_to_a_commit(alias):
    entry = MODEL_REGISTRY[alias]
    assert re.fullmatch(r"[0-9a-f]{40}", entry.revision or "")
    with patch("speakeronnx.embedder.hf_hub_download", return_value="/fake.onnx") as download:
        entry.download()
    download.assert_called_once_with(repo_id=entry.hf_repo, filename=entry.hf_file, revision=entry.revision)
