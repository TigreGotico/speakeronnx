"""Audio loading tests — edge cases, all bit depths, stereo, resampling."""
import math
import os
import struct
import tempfile
import wave
import unittest

import numpy as np


_8BIT_SINE = bytes([
    int(128 + 127 * math.sin(2 * math.pi * 440 * t / 16000))
    for t in range(1600)
])


def _write_wav(path, samples, sample_width=2, framerate=16000, n_channels=1):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(framerate)
        wf.writeframes(samples)


class TestLoadAudio(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _path(self, name):
        return os.path.join(self.tmpdir, name)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir)

    # --- PCM bit depths ---

    def test_8bit_wav(self):
        from speakeronnx.embedder import load_audio
        path = self._path("test8.wav")
        _write_wav(path, _8BIT_SINE, sample_width=1)
        audio = load_audio(path, target_sr=16000)
        self.assertEqual(audio.ndim, 1)
        self.assertGreater(len(audio), 0)

    def test_16bit_wav(self):
        from speakeronnx.embedder import load_audio
        path = self._path("test16.wav")
        raw = struct.pack("<%dh" % 1600,
                         *[int(32767 * math.sin(2 * math.pi * 440 * t / 16000))
                           for t in range(1600)])
        _write_wav(path, raw, sample_width=2)
        audio = load_audio(path, target_sr=16000)
        self.assertAlmostEqual(float(np.max(np.abs(audio))), 1.0, delta=0.001)

    def test_24bit_wav(self):
        import struct
        from speakeronnx.embedder import load_audio
        path = self._path("test24.wav")
        samples_24 = b""
        for t in range(400):
            v = int(8388607 * math.sin(2 * math.pi * 440 * t / 16000))
            v = max(-8388608, min(8388607, v))
            samples_24 += struct.pack("<i", v)[:3]
        _write_wav(path, samples_24, sample_width=3, framerate=16000)
        audio = load_audio(path, target_sr=16000)
        self.assertAlmostEqual(float(np.max(np.abs(audio))), 1.0, delta=0.01)

    def test_32bit_int_wav(self):
        from speakeronnx.embedder import load_audio
        path = self._path("test32i.wav")
        raw = struct.pack("<%di" % 400,
                         *[int(2147483647 * math.sin(2 * math.pi * 440 * t / 16000))
                           for t in range(400)])
        _write_wav(path, raw, sample_width=4, framerate=16000)
        audio = load_audio(path, target_sr=16000)
        self.assertAlmostEqual(float(np.max(np.abs(audio))), 1.0, delta=0.01)

    def test_32bit_float_wav(self):
        from speakeronnx.embedder import load_audio
        path = self._path("test32f.wav")
        raw = struct.pack("<%df" % 400,
                         *[0.5 * math.sin(2 * math.pi * 440 * t / 16000)
                           for t in range(400)])
        _write_wav(path, raw, sample_width=4, framerate=16000)
        audio = load_audio(path, target_sr=16000)
        self.assertAlmostEqual(float(np.max(np.abs(audio))), 0.5, delta=0.01)

    # --- Stereo downmix ---

    def test_stereo_downmix(self):
        from speakeronnx.embedder import load_audio
        path = self._path("stereo.wav")
        n = 1600
        left = [int(10000 * math.sin(2 * math.pi * 440 * t / 16000)) for t in range(n)]
        right = [int(10000 * math.sin(2 * math.pi * 880 * t / 16000)) for t in range(n)]
        interleaved = struct.pack("<%dh" % (2 * n), *[v for pair in zip(left, right) for v in pair])
        _write_wav(path, interleaved, sample_width=2, framerate=16000, n_channels=2)
        audio = load_audio(path, target_sr=16000)
        self.assertEqual(audio.ndim, 1)

    # --- Resampling ---

    def test_resample_48k_to_16k(self):
        from speakeronnx.embedder import load_audio
        path = self._path("resamp.wav")
        sr = 48000
        n = int(sr * 0.5)
        raw = struct.pack("<%dh" % n,
                         *[int(10000 * math.sin(2 * math.pi * 440 * t / sr))
                           for t in range(n)])
        _write_wav(path, raw, sample_width=2, framerate=sr)
        audio = load_audio(path, target_sr=16000)
        expected_len = int(n * 16000 / sr)
        self.assertAlmostEqual(len(audio), expected_len, delta=2)

    def test_resample_8k_to_16k(self):
        from speakeronnx.embedder import load_audio
        path = self._path("resamp8k.wav")
        sr = 8000
        n = int(sr * 0.5)
        raw = struct.pack("<%dh" % n,
                         *[int(10000 * math.sin(2 * math.pi * 440 * t / sr))
                           for t in range(n)])
        _write_wav(path, raw, sample_width=2, framerate=sr)
        audio = load_audio(path, target_sr=16000)
        expected_len = int(n * 16000 / sr)
        self.assertAlmostEqual(len(audio), expected_len, delta=2)

    def test_unsupported_sample_width_raises(self):
        from speakeronnx.embedder import _load_wav
        path = self._path("bad.wav")
        # Write a raw WAV with sample width=6 (not standard) using direct bytes
        import struct as _struct
        n_channels = 1
        sample_width = 6
        framerate = 16000
        data = b"\x00" * 60
        data_size = len(data)
        with open(path, "wb") as f:
            f.write(b"RIFF")
            f.write(_struct.pack("<I", 36 + data_size))
            f.write(b"WAVE")
            f.write(b"fmt ")
            f.write(_struct.pack("<I", 16))
            f.write(_struct.pack("<H", 1))
            f.write(_struct.pack("<H", n_channels))
            f.write(_struct.pack("<I", framerate))
            f.write(_struct.pack("<I", framerate * n_channels * sample_width))
            f.write(_struct.pack("<H", n_channels * sample_width))
            f.write(_struct.pack("<H", sample_width * 8))
            f.write(b"data")
            f.write(_struct.pack("<I", data_size))
            f.write(data)
        with self.assertRaises(ValueError):
            _load_wav(path)

    def test_silence_returns_zeros(self):
        from speakeronnx.embedder import load_audio
        path = self._path("silence.wav")
        raw = struct.pack("<%dh" % 1600, *([0] * 1600))
        _write_wav(path, raw, sample_width=2)
        audio = load_audio(path, target_sr=16000)
        np.testing.assert_array_equal(audio, np.zeros(1600, dtype=np.float32))

    def test_empty_wav_returns_empty(self):
        from speakeronnx.embedder import load_audio
        path = self._path("empty.wav")
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"")
        audio = load_audio(path, target_sr=16000)
        self.assertEqual(len(audio), 0)


import numpy as np
