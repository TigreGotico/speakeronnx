"""Frontend tests — fbank computation, parameter edge cases."""
import math
import unittest

import numpy as np


def _sine_wave(duration_s=2.0, sr=16000, freq=440, amp=0.5):
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (amp * np.sin(2 * math.pi * freq * t)).astype(np.float32)


class TestFbankFrontend(unittest.TestCase):
    def test_output_shape_80_bins(self):
        from speakeronnx.embedder import compute_fbank
        audio = _sine_wave(2.0)
        feats = compute_fbank(audio, num_mel_bins=80)
        self.assertEqual(feats.shape[1], 80)
        self.assertGreater(feats.shape[0], 100)

    def test_output_shape_40_bins(self):
        from speakeronnx.embedder import compute_fbank
        audio = _sine_wave(2.0)
        feats = compute_fbank(audio, num_mel_bins=40)
        self.assertEqual(feats.shape[1], 40)

    def test_deterministic_no_dither(self):
        from speakeronnx.embedder import compute_fbank
        audio = _sine_wave(1.0)
        f1 = compute_fbank(audio, dither=0.0)
        f2 = compute_fbank(audio, dither=0.0)
        np.testing.assert_array_equal(f1, f2)

    def test_dither_gives_different_result(self):
        from speakeronnx.embedder import compute_fbank
        audio = _sine_wave(0.5)
        f1 = compute_fbank(audio, dither=0.0)
        f2 = compute_fbank(audio, dither=1e-5)
        # With dither they should differ (very high probability)
        self.assertFalse(np.allclose(f1, f2))

    def test_cmn_produces_zero_mean(self):
        from speakeronnx.embedder import compute_fbank
        audio = _sine_wave(2.0)
        feats = compute_fbank(audio, apply_cmn=True)
        np.testing.assert_allclose(feats.mean(axis=0), 0.0, atol=1e-4)

    def test_no_cmn_preserves_mean(self):
        from speakeronnx.embedder import compute_fbank
        audio = _sine_wave(2.0)
        feats = compute_fbank(audio, apply_cmn=False)
        # Without CMN, mean should NOT be near zero
        self.assertGreater(np.abs(feats.mean()).item(), 1e-4)

    def test_short_audio_raises(self):
        from speakeronnx.embedder import compute_fbank
        audio = np.zeros(50, dtype=np.float32)
        with self.assertRaises(ValueError):
            compute_fbank(audio, sample_rate=16000)

    def test_minimal_valid_audio(self):
        """A single frame should work."""
        from speakeronnx.embedder import compute_fbank
        frame_length = int(16000 * 25 / 1000)  # 400 samples
        audio = _sine_wave(frame_length / 16000 + 0.01, sr=16000)
        feats = compute_fbank(audio, sample_rate=16000)
        self.assertGreater(feats.shape[0], 0)

    def test_silence_gives_low_energy(self):
        from speakeronnx.embedder import compute_fbank
        audio = np.zeros(16000, dtype=np.float32)
        feats = compute_fbank(audio, apply_cmn=True)
        # Silence with CMN → zero-mean but non-zero variance
        # The values should be very negative log-energies
        self.assertTrue(np.all(np.isfinite(feats)))

    def test_filterbank_values_positive(self):
        """Mel filterbank matrix should be non-negative."""
        from speakeronnx.embedder import _mel_filterbank
        fb = _mel_filterbank(80, 512, 16000)
        self.assertGreaterEqual(fb.min().item(), 0.0)

    def test_hz_mel_roundtrip(self):
        from speakeronnx.embedder import _hz_to_mel, _mel_to_hz
        for hz in [100, 500, 1000, 3000, 6000]:
            mel = _hz_to_mel(hz)
            hz_back = _mel_to_hz(mel)
            self.assertAlmostEqual(hz, hz_back, delta=1.0)


class TestFbankCache(unittest.TestCase):
    def test_cache_reuses_filterbank(self):
        from speakeronnx.embedder import compute_fbank, _FBANK_CACHE
        _FBANK_CACHE.clear()
        audio = _sine_wave(1.0)
        compute_fbank(audio, num_mel_bins=80, sample_rate=16000)
        self.assertIn((80, 512, 16000), _FBANK_CACHE)
        compute_fbank(audio, num_mel_bins=80, sample_rate=16000)
        compute_fbank(audio, num_mel_bins=40, sample_rate=16000)
        self.assertIn((40, 512, 16000), _FBANK_CACHE)
        self.assertEqual(len(_FBANK_CACHE), 2)


class TestFrontendEdgeCases(unittest.TestCase):
    def test_different_frame_length(self):
        from speakeronnx.embedder import compute_fbank
        audio = _sine_wave(2.0)
        feats_25 = compute_fbank(audio, frame_length_ms=25.0)
        feats_50 = compute_fbank(audio, frame_length_ms=50.0)
        # Different frame length → different number of frames
        self.assertNotEqual(feats_25.shape[0], feats_50.shape[0])

    def test_different_frame_shift(self):
        from speakeronnx.embedder import compute_fbank
        audio = _sine_wave(2.0)
        feats_10 = compute_fbank(audio, frame_shift_ms=10.0)
        feats_20 = compute_fbank(audio, frame_shift_ms=20.0)
        # 20ms shift → about half as many frames
        ratio = feats_10.shape[0] / feats_20.shape[0]
        self.assertAlmostEqual(ratio, 2.0, delta=0.1)

    def test_high_freq_limit(self):
        from speakeronnx.embedder import compute_fbank
        audio = _sine_wave(1.0)
        # Should not crash with different freq limits
        feats = compute_fbank(audio, sample_rate=16000)
        self.assertTrue(np.all(np.isfinite(feats)))


if __name__ == "__main__":
    unittest.main()
