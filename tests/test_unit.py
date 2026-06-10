"""Unit tests — no real model download required (sessions are mocked)."""
import math
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


class TestModelRegistry(unittest.TestCase):
    def test_registry_has_expected_aliases(self):
        from speakeronnx import MODEL_REGISTRY
        self.assertIn("wespeaker-resnet34", MODEL_REGISTRY)
        self.assertIn("wespeaker-ecapa512", MODEL_REGISTRY)
        self.assertIn("wespeaker-resnet293", MODEL_REGISTRY)
        self.assertIn("campplus", MODEL_REGISTRY)
        self.assertIn("campplus-zh-en", MODEL_REGISTRY)
        self.assertIn("eres2net", MODEL_REGISTRY)
        self.assertIn("titanet-small", MODEL_REGISTRY)
        self.assertIn("titanet-large", MODEL_REGISTRY)
        self.assertIn("redimnet-b2", MODEL_REGISTRY)

    def test_registry_entries_have_required_fields(self):
        from speakeronnx import MODEL_REGISTRY
        for alias, entry in MODEL_REGISTRY.items():
            self.assertTrue(entry.hf_repo, f"{alias} missing hf_repo")
            self.assertTrue(entry.hf_file, f"{alias} missing hf_file")
            self.assertTrue(entry.license, f"{alias} missing license")
            self.assertGreater(entry.embed_dim, 0, f"{alias} embed_dim <= 0")
            self.assertIn(entry.sample_rate, (8000, 16000, 22050, 24000, 44100, 48000))
            self.assertIn(entry.frontend, ("fbank80", "raw"))

    def test_model_entry_alias_matches_key(self):
        from speakeronnx import MODEL_REGISTRY
        for alias, entry in MODEL_REGISTRY.items():
            self.assertEqual(alias, entry.alias)


class TestCosineAndVerify(unittest.TestCase):
    def test_cosine_identical(self):
        from speakeronnx import cosine
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.assertAlmostEqual(cosine(v, v), 1.0, places=6)

    def test_cosine_orthogonal(self):
        from speakeronnx import cosine
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        self.assertAlmostEqual(cosine(a, b), 0.0, places=6)

    def test_cosine_opposite(self):
        from speakeronnx import cosine
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        self.assertAlmostEqual(cosine(v, -v), -1.0, places=6)

    def test_cosine_zero_vector(self):
        from speakeronnx import cosine
        z = np.zeros(3, dtype=np.float32)
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.assertEqual(cosine(z, v), 0.0)

    def test_verify_accepts_above_threshold(self):
        from speakeronnx import verify
        v = np.array([1.0, 0.0], dtype=np.float32)
        ok, score = verify(v, v, threshold=0.5)
        self.assertTrue(ok)
        self.assertAlmostEqual(score, 1.0, places=6)

    def test_verify_rejects_below_threshold(self):
        from speakeronnx import verify
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        ok, score = verify(a, b, threshold=0.5)
        self.assertFalse(ok)
        self.assertAlmostEqual(score, 0.0, places=6)


class TestFbankFrontend(unittest.TestCase):
    def _make_audio(self, duration_s: float = 2.0, sr: int = 16000) -> np.ndarray:
        t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
        return (0.5 * np.sin(2 * math.pi * 440 * t)).astype(np.float32)

    def test_fbank_output_shape(self):
        from speakeronnx.embedder import compute_fbank
        audio = self._make_audio(2.0)
        feats = compute_fbank(audio, sample_rate=16000, num_mel_bins=80)
        # T depends on framing: ~(2.0s * 16000 - 400) / 160 ≈ 197 frames
        self.assertEqual(feats.shape[1], 80)
        self.assertGreater(feats.shape[0], 10)

    def test_fbank_deterministic(self):
        from speakeronnx.embedder import compute_fbank
        audio = self._make_audio(1.0)
        f1 = compute_fbank(audio, dither=0.0)
        f2 = compute_fbank(audio, dither=0.0)
        np.testing.assert_array_equal(f1, f2)

    def test_fbank_cmn_near_zero_mean(self):
        from speakeronnx.embedder import compute_fbank
        audio = self._make_audio(2.0)
        feats = compute_fbank(audio, apply_cmn=True)
        # Per-utterance CMN means the time-axis mean of each mel bin is ~0
        np.testing.assert_allclose(feats.mean(axis=0), 0.0, atol=1e-4)

    def test_fbank_short_audio_raises(self):
        from speakeronnx.embedder import compute_fbank
        audio = np.zeros(100, dtype=np.float32)  # far too short
        with self.assertRaises(ValueError):
            compute_fbank(audio, sample_rate=16000)


class TestAudioLoading(unittest.TestCase):
    def test_load_numpy_array_passthrough(self):
        from speakeronnx.embedder import load_audio
        arr = np.random.randn(16000).astype(np.float32)
        out = load_audio(arr, target_sr=16000)
        np.testing.assert_array_equal(arr, out)

    def test_load_wav_16bit(self):
        """Write a tiny 16-bit WAV and read it back."""
        import io, wave, tempfile, os
        audio_in = np.array([0, 1000, -1000, 32767, -32768], dtype=np.int16)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            with wave.open(tmp, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_in.tobytes())
            from speakeronnx.embedder import load_audio
            out = load_audio(tmp, target_sr=16000)
            self.assertEqual(len(out), len(audio_in))
            # peak should be ~1.0
            self.assertAlmostEqual(float(np.max(np.abs(out))), 1.0, delta=0.001)
        finally:
            os.unlink(tmp)


class TestSpeakerEmbedderMocked(unittest.TestCase):
    """Test SpeakerEmbedder with a mocked ONNX session — no download."""

    def _make_embedder_with_mock(self, alias: str = "wespeaker-resnet34"):
        """Return a SpeakerEmbedder with download + session mocked."""
        from speakeronnx.embedder import SpeakerEmbedder, MODEL_REGISTRY
        entry = MODEL_REGISTRY[alias]
        fake_emb = np.random.randn(entry.embed_dim).astype(np.float32)
        fake_emb /= np.linalg.norm(fake_emb)

        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="feats")]
        mock_session.get_outputs.return_value = [MagicMock(name="emb")]
        mock_session.run.return_value = [fake_emb]

        with patch("speakeronnx.embedder.hf_hub_download", return_value="/fake/model.onnx"), \
             patch("onnxruntime.InferenceSession", return_value=mock_session):
            embedder = SpeakerEmbedder(model=alias)
        embedder._session = mock_session
        return embedder, fake_emb

    def _make_wav(self, sr: int = 16000, duration: float = 2.0) -> "str":
        import wave, tempfile, os
        audio = (np.sin(np.linspace(0, 440 * 2 * np.pi * duration, int(sr * duration)))
                 * 10000).astype(np.int16)
        f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp = f.name
        f.close()
        with wave.open(tmp, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(audio.tobytes())
        return tmp

    def test_embed_returns_l2_normalised(self):
        import os
        embedder, _ = self._make_embedder_with_mock()
        tmp = self._make_wav()
        try:
            emb = embedder.embed(tmp)
            self.assertAlmostEqual(float(np.linalg.norm(emb)), 1.0, delta=1e-5)
        finally:
            os.unlink(tmp)

    def test_embed_shape(self):
        import os
        from speakeronnx.embedder import MODEL_REGISTRY
        alias = "wespeaker-resnet34"
        embedder, _ = self._make_embedder_with_mock(alias)
        tmp = self._make_wav()
        try:
            emb = embedder.embed(tmp)
            # mock returns embed_dim-dim vector
            self.assertEqual(emb.ndim, 1)
        finally:
            os.unlink(tmp)

    def test_cosine_same_embedding(self):
        embedder, fake = self._make_embedder_with_mock()
        score = embedder.cosine(fake, fake)
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_verify_threshold(self):
        embedder, fake = self._make_embedder_with_mock()
        ok, score = embedder.verify(fake, fake, threshold=0.5)
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
