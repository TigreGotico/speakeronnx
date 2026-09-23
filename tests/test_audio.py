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


def _write_raw_wav(path, data, tag, sample_width, framerate=16000, n_channels=1, extensible=False):
    """Write a WAV header by hand, for the formats the stdlib wave writer cannot produce."""
    block = n_channels * sample_width
    if extensible:
        fmt = struct.pack("<HHIIHHHHI", 0xFFFE, n_channels, framerate, framerate * block, block,
                          sample_width * 8, 22, sample_width * 8, 0)
        fmt += struct.pack("<H", tag) + b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
    else:
        fmt = struct.pack("<HHIIHH", tag, n_channels, framerate, framerate * block, block, sample_width * 8)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", len(body)) + body)


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

    # --- IEEE float and format tags ---

    def test_float32_wav(self):
        from speakeronnx.embedder import load_audio
        path = self._path("f32.wav")
        x = (0.5 * np.sin(2 * np.pi * 440 * np.arange(1600) / 16000)).astype("<f4")
        _write_raw_wav(path, x.tobytes(), tag=3, sample_width=4)
        np.testing.assert_allclose(load_audio(path, target_sr=16000), x, atol=1e-7)

    def test_float64_wav(self):
        from speakeronnx.embedder import load_audio
        path = self._path("f64.wav")
        x = 0.25 * np.cos(2 * np.pi * 220 * np.arange(1600) / 16000)
        _write_raw_wav(path, x.astype("<f8").tobytes(), tag=3, sample_width=8)
        np.testing.assert_allclose(load_audio(path, target_sr=16000), x.astype(np.float32), atol=1e-7)

    def test_extensible_float32_wav(self):
        from speakeronnx.embedder import load_audio
        path = self._path("f32x.wav")
        x = np.linspace(-0.9, 0.9, 1600, dtype="<f4")
        _write_raw_wav(path, x.tobytes(), tag=3, sample_width=4, extensible=True)
        np.testing.assert_allclose(load_audio(path, target_sr=16000), x, atol=1e-7)

    def test_unknown_format_tag_raises(self):
        from speakeronnx.embedder import _load_wav
        path = self._path("alaw.wav")
        _write_raw_wav(path, b"\x00" * 100, tag=6, sample_width=1)
        with self.assertRaises(ValueError):
            _load_wav(path)

    def test_chunk_before_fmt_is_skipped(self):
        from speakeronnx.embedder import load_audio
        path = self._path("list.wav")
        x = (np.arange(1600) % 100 - 50).astype("<i2")
        _write_raw_wav(path, x.tobytes(), tag=1, sample_width=2)
        raw = open(path, "rb").read()
        extra = b"LIST" + struct.pack("<I", 5) + b"abcde\x00"
        body = raw[8:12] + extra + raw[12:]
        with open(path, "wb") as f:
            f.write(b"RIFF" + struct.pack("<I", len(body)) + body)
        np.testing.assert_allclose(load_audio(path, target_sr=16000), x.astype(np.float32) / 32768.0)

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
