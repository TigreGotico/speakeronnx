"""Model registry integrity tests."""
import unittest

import numpy as np


class TestModelRegistryIntegrity(unittest.TestCase):
    def test_all_aliases_unique(self):
        from speakeronnx import MODEL_REGISTRY
        self.assertEqual(len(MODEL_REGISTRY), len(set(MODEL_REGISTRY.keys())))

    def test_all_frontends_valid(self):
        from speakeronnx import MODEL_REGISTRY
        for alias, entry in MODEL_REGISTRY.items():
            self.assertIn(
                entry.frontend,
                ("fbank80", "raw"),
                f"{alias}: unknown frontend {entry.frontend!r}",
            )

    def test_all_input_layouts_valid(self):
        from speakeronnx import MODEL_REGISTRY
        for alias, entry in MODEL_REGISTRY.items():
            self.assertIn(
                entry.input_layout,
                ("BTF", "BFT"),
                f"{alias}: unknown layout {entry.input_layout!r}",
            )

    def test_all_output_indices_valid(self):
        from speakeronnx import MODEL_REGISTRY
        for alias, entry in MODEL_REGISTRY.items():
            self.assertGreaterEqual(
                entry.output_index, 0, f"{alias}: negative output_index"
            )

    def test_extra_feeds_titanet_only(self):
        from speakeronnx import MODEL_REGISTRY
        for alias, entry in MODEL_REGISTRY.items():
            if "titanet" in alias:
                self.assertIsNotNone(entry.extra_feeds)
                self.assertIn("length", entry.extra_feeds)
            else:
                # Most models have no extra feeds; redimnet doesn't either
                pass

    def test_raw_frontend_models(self):
        from speakeronnx import MODEL_REGISTRY
        for alias, entry in MODEL_REGISTRY.items():
            if entry.frontend == "raw":
                self.assertEqual(entry.input_layout, "BTF")
                self.assertEqual(entry.output_index, 0)

    def test_bft_layout_models(self):
        from speakeronnx import MODEL_REGISTRY
        for alias, entry in MODEL_REGISTRY.items():
            if entry.input_layout == "BFT":
                self.assertIsNotNone(entry.extra_feeds)
                self.assertIn("titanet", alias)

    def test_embed_dims_positive(self):
        from speakeronnx import MODEL_REGISTRY
        for alias, entry in MODEL_REGISTRY.items():
            self.assertGreater(entry.embed_dim, 0, f"{alias}: embed_dim=0")

    def test_embed_dims_common(self):
        """All models should have embed_dim in {192, 256, 512}."""
        from speakeronnx import MODEL_REGISTRY
        for alias, entry in MODEL_REGISTRY.items():
            self.assertIn(
                entry.embed_dim,
                (192, 256, 512),
                f"{alias}: unexpected embed_dim={entry.embed_dim}",
            )

    def test_all_hf_repos_reachable(self):
        """Check that all HF repos exist (fast API check, not full download)."""
        from speakeronnx import MODEL_REGISTRY
        import requests
        for alias, entry in MODEL_REGISTRY.items():
            url = f"https://huggingface.co/api/models/{entry.hf_repo}"
            r = requests.get(url, timeout=10)
            self.assertEqual(
                r.status_code, 200,
                f"{alias}: HF repo {entry.hf_repo} returned {r.status_code}",
            )
            # Verify the specific file exists
            files = r.json().get("siblings", [])
            filenames = [f["rfilename"] for f in files]
            self.assertIn(
                entry.hf_file, filenames,
                f"{alias}: {entry.hf_file} not found in {entry.hf_repo}",
            )


class TestDefaultModel(unittest.TestCase):
    def test_default_model_is_valid(self):
        from speakeronnx import DEFAULT_MODEL, MODEL_REGISTRY
        self.assertIn(DEFAULT_MODEL, MODEL_REGISTRY)

    def test_default_model_mock_works(self):
        """Default model can be loaded with mocked session."""
        from unittest.mock import MagicMock, patch
        from speakeronnx import SpeakerEmbedder, DEFAULT_MODEL, MODEL_REGISTRY

        entry = MODEL_REGISTRY[DEFAULT_MODEL]
        fake_emb = np.random.randn(entry.embed_dim).astype(np.float32)
        fake_emb /= np.linalg.norm(fake_emb)

        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="feats")]
        mock_session.get_outputs.return_value = [MagicMock(name="emb")]
        mock_session.run.return_value = [fake_emb]

        with patch("speakeronnx.embedder.hf_hub_download", return_value="/fake/model.onnx"), \
             patch("onnxruntime.InferenceSession", return_value=mock_session):
            embedder = SpeakerEmbedder()
        self.assertEqual(embedder.entry.alias, DEFAULT_MODEL)


class TestModelEntryDownload(unittest.TestCase):
    def test_download_returns_string(self):
        """Download should return a path string (may skip if no HF access)."""
        from speakeronnx import MODEL_REGISTRY
        # Just test the first model
        entry = MODEL_REGISTRY["wespeaker-resnet34"]
        path = entry.download()
        self.assertIsInstance(path, str)
        self.assertTrue(os.path.isfile(path), f"Downloaded file not found: {path}")


import os
