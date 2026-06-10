"""SpeakerEmbedder tests — mocked ONNX sessions, verifying input shaping."""
from unittest.mock import MagicMock, patch
import numpy as np


class TestEmbedderInputShaping:
    """Test that embed() shapes inputs correctly for each model type."""

    @patch("speakeronnx.embedder.hf_hub_download", return_value="/fake/model.onnx")
    @patch("onnxruntime.InferenceSession")
    def test_fbank80_btf_layout(self, mock_ort, mock_hf):
        """WeSpeaker models: fbank80 → [1, T, 80] (BTF layout)."""
        from speakeronnx import SpeakerEmbedder
        entry = SpeakerEmbedder(model="wespeaker-resnet34")
        # Capture the input that gets fed to ONNX
        entry._session.run = MagicMock(
            return_value=[np.random.randn(256).astype(np.float32)]
        )
        entry.embed(np.random.randn(16000).astype(np.float32))
        feed = entry._session.run.call_args[0][1]
        input_name = entry._session.get_inputs()[0].name
        inp = feed[input_name]
        assert inp.ndim == 3, f"Expected 3D input, got {inp.ndim}D"
        # BTF: (B=1, T, F=80)
        assert inp.shape[0] == 1, f"Batch dim should be 1, got {inp.shape[0]}"
        assert inp.shape[2] == 80, f"Feature dim should be 80, got {inp.shape[2]}"

    @patch("speakeronnx.embedder.hf_hub_download", return_value="/fake/model.onnx")
    @patch("onnxruntime.InferenceSession")
    def test_titanet_bft_layout(self, mock_ort, mock_hf):
        """TitaNet models: fbank80 → [1, 80, T] (BFT layout)."""
        from speakeronnx import SpeakerEmbedder
        entry = SpeakerEmbedder(model="titanet-small")
        entry._session.run = MagicMock(
            return_value=[np.random.randn(192).astype(np.float32)]
        )
        entry.embed(np.random.randn(16000).astype(np.float32))
        feed = entry._session.run.call_args[0][1]
        input_name = entry._session.get_inputs()[0].name
        inp = feed[input_name]
        assert inp.ndim == 3, f"Expected 3D input, got {inp.ndim}D"
        # BFT: (B=1, F=80, T)
        assert inp.shape[0] == 1, f"Batch dim should be 1, got {inp.shape[0]}"
        assert inp.shape[1] == 80, f"Feature dim should be 80, got {inp.shape[1]}"

    @patch("speakeronnx.embedder.hf_hub_download", return_value="/fake/model.onnx")
    @patch("onnxruntime.InferenceSession")
    def test_titanet_extra_feed_length(self, mock_ort, mock_hf):
        """TitaNet should include a 'length' feed equal to frame count."""
        from speakeronnx import SpeakerEmbedder
        entry = SpeakerEmbedder(model="titanet-small")
        mock_run = MagicMock(return_value=[np.random.randn(192).astype(np.float32)])
        entry._session.run = mock_run
        entry.embed(np.random.randn(16000).astype(np.float32))
        feed = mock_run.call_args[0][1]
        assert "length" in feed, "TitaNet feed missing 'length' tensor"
        assert feed["length"].dtype == np.int64

    @patch("speakeronnx.embedder.hf_hub_download", return_value="/fake/model.onnx")
    @patch("onnxruntime.InferenceSession")
    def test_titanet_output_index_1(self, mock_ort, mock_hf):
        """TitaNet should use output index 1 (embs, not logits)."""
        from speakeronnx import SpeakerEmbedder
        entry = SpeakerEmbedder(model="titanet-small")
        assert entry._output_name == entry._session.get_outputs()[1].name

    @patch("speakeronnx.embedder.hf_hub_download", return_value="/fake/model.onnx")
    @patch("onnxruntime.InferenceSession")
    def test_raw_frontend_input_shape(self, mock_ort, mock_hf):
        """redimnet-b2: raw audio → [1, 1, T] waveform."""
        from speakeronnx import SpeakerEmbedder
        entry = SpeakerEmbedder(model="redimnet-b2")
        entry._session.run = MagicMock(
            return_value=[np.random.randn(192).astype(np.float32)]
        )
        entry.embed(np.random.randn(16000).astype(np.float32))
        feed = entry._session.run.call_args[0][1]
        input_name = entry._session.get_inputs()[0].name
        inp = feed[input_name]
        assert inp.ndim == 3, f"Expected 3D input, got {inp.ndim}D"
        assert inp.shape[1] == 1, f"Channel dim should be 1, got {inp.shape[1]}"


class TestEmbedderEdgeCases:
    def test_model_not_found_raises(self):
        from speakeronnx.embedder import SpeakerEmbedder
        try:
            SpeakerEmbedder(model="nonexistent-model")
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass

    def test_file_not_found_raises(self):
        from speakeronnx.embedder import SpeakerEmbedder
        try:
            SpeakerEmbedder(model="/nonexistent/path.onnx")
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass
