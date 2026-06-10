"""CLI tests — mocked ONNX sessions, no real audio needed."""
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


class TestCLI(unittest.TestCase):
    def _mock_embedder(self, dim=256):
        """Patch SpeakerEmbedder to return a fake embedding."""
        fake_emb = np.random.randn(dim).astype(np.float32)
        fake_emb /= np.linalg.norm(fake_emb)

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = fake_emb
        mock_embedder.sample_rate = 16000
        mock_embedder.embed_dim = dim
        mock_embedder.entry = MagicMock()
        mock_embedder.entry.alias = "wespeaker-resnet34"
        mock_embedder.entry.embed_dim = dim
        return mock_embedder

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.fake_wav = os.path.join(self.tmpdir, "test.wav")
        # Write a minimal valid WAV header + data
        import wave
        import struct
        with wave.open(self.fake_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(struct.pack("<%dh" % 1600, *([0] * 1600)))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir)

    @patch("speakeronnx.SpeakerEmbedder")
    def test_cli_embed_default(self, mock_cls):
        mock_cls.return_value = self._mock_embedder()
        from speakeronnx.__main__ import main

        test_args = ["speakeronnx", "embed", self.fake_wav]
        with patch.object(sys, "argv", test_args):
            main()

        mock_cls.assert_called_once_with(model="wespeaker-resnet34")

    @patch("speakeronnx.SpeakerEmbedder")
    def test_cli_embed_with_model(self, mock_cls):
        mock_cls.return_value = self._mock_embedder(192)
        from speakeronnx.__main__ import main

        test_args = ["speakeronnx", "embed", self.fake_wav, "--model", "wespeaker-ecapa512"]
        with patch.object(sys, "argv", test_args):
            main()

        mock_cls.assert_called_once_with(model="wespeaker-ecapa512")

    @patch("speakeronnx.SpeakerEmbedder")
    def test_cli_verify_same(self, mock_cls):
        """verify exits 0 when score >= threshold."""
        mock = self._mock_embedder()
        mock.verify.return_value = (True, 0.85)
        mock_cls.return_value = mock

        from speakeronnx.__main__ import main
        test_args = ["speakeronnx", "verify", self.fake_wav, self.fake_wav]
        with patch.object(sys, "argv", test_args):
            with self.assertRaises(SystemExit) as ctx:
                main()
        self.assertEqual(ctx.exception.code, 0)

    @patch("speakeronnx.SpeakerEmbedder")
    def test_cli_verify_different(self, mock_cls):
        """verify exits 1 when score < threshold."""
        mock = self._mock_embedder()
        mock.verify.return_value = (False, 0.15)
        mock_cls.return_value = mock

        from speakeronnx.__main__ import main
        test_args = ["speakeronnx", "verify", self.fake_wav, self.fake_wav]
        with patch.object(sys, "argv", test_args):
            with self.assertRaises(SystemExit) as ctx:
                main()
        self.assertEqual(ctx.exception.code, 1)

    @patch("speakeronnx.MODEL_REGISTRY")
    def test_cli_list(self, mock_registry):
        """list prints all models without error."""
        from speakeronnx import ModelEntry
        mock_registry.items.return_value = [
            ("model-a", ModelEntry(
                alias="model-a", hf_repo="r", hf_file="f.onnx",
                license="mit", embed_dim=192, sample_rate=16000,
                frontend="fbank80", description="desc",
            )),
        ]
        from speakeronnx.__main__ import main
        test_args = ["speakeronnx", "list"]
        with patch.object(sys, "argv", test_args):
            main()  # should not raise


if __name__ == "__main__":
    unittest.main()
