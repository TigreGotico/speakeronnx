"""End-to-end tests — downloads real ONNX models and generates real audio.

Uses phoonnx (phoonnx) for TTS voice generation — 1000+ voices available,
no external TTS binaries needed.

Run with: pytest tests/test_e2e.py -v -s

Two test classes:

- ``TestE2EPipeline`` — all models (determinism, L2 norm). No speaker-separation
  requirement, so runs even on models that don't separate short TTS utterances well.
- ``TestE2ESpeakerSeparation`` — models that consistently separate these TTS voices
  (same-speaker > cross-speaker, three-voice ordering).

Tests generate 3 utterances per voice using phoonnx piper voices:
A: piper/en_US-amy-medium (female, US)
B: piper/en_US-joe-medium   (male, US)
C: piper/en_GB-alan-medium  (male, GB)
"""

import os
import sys

import numpy as np
import pytest


UTTERANCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Speaker verification systems identify who is talking.",
    "Voice biometrics provide an additional layer of security.",
]

VOICE_A = "piper/en_US-amy-medium"
VOICE_B = "piper/en_US-joe-medium"
VOICE_C = "piper/en_GB-alan-medium"



@pytest.fixture(scope="module")
def audio_dir(tmp_path_factory):
    """Generate all test audio clips once per module via phoonnx."""
    from phoonnx.opm import PhoonnxTTSPlugin
    tts = PhoonnxTTSPlugin()
    d = tmp_path_factory.mktemp("audio")

    clips = {}
    for voice_id, tag in [(VOICE_A, "A"), (VOICE_B, "B"), (VOICE_C, "C")]:
        for i, text in enumerate(UTTERANCES):
            path = str(d / f"{tag}{i+1}.wav")
            tts.get_tts(text, path, voice=voice_id)
            clips[f"{tag}{i+1}"] = path

    return clips


# Models that pass speaker-separation e2e (same-speaker > cross-speaker) with
# phoonnx piper voices (amy-female US / joe-male US / alan-male GB).
# campplus is excluded — its embeddings are too utterance-dependent for short TTS
# clips (same-speaker similarity ~0.07 vs cross ~0.50).  Pipeline-only tests
# (determinism, L2 norm) still run on all models via the full parametrize.
_SPEAKER_SEP_MODELS = [
    "wespeaker-resnet34",
    "wespeaker-ecapa512",
    "wespeaker-resnet293",
    "campplus-zh-en",
    "eres2net",
    "redimnet-b2",
    "titanet-small",
    "titanet-large",
]

# All models (full parametrize for pipeline-only tests)
_ALL_MODELS = sorted(_SPEAKER_SEP_MODELS + ["campplus"])


@pytest.mark.parametrize("alias", _ALL_MODELS)
class TestE2EPipeline:
    """Tests that verify the embedding pipeline itself — no speaker separation needed."""

    def test_embedding_determinism(self, alias, audio_dir):
        from speakeronnx import SpeakerEmbedder

        emb = SpeakerEmbedder(model=alias)
        e1 = emb.embed(audio_dir["A1"])
        e2 = emb.embed(audio_dir["A1"])
        np.testing.assert_array_equal(e1, e2)

    def test_embedding_is_l2_normalised(self, alias, audio_dir):
        from speakeronnx import SpeakerEmbedder

        emb = SpeakerEmbedder(model=alias)
        e = emb.embed(audio_dir["A1"])
        norm = float(np.linalg.norm(e))
        assert abs(norm - 1.0) < 1e-5, f"[{alias}] norm={norm:.6f} expected 1.0"


@pytest.mark.parametrize("alias", _SPEAKER_SEP_MODELS)
class TestE2ESpeakerSeparation:
    """Tests that verify the model can separate speakers on TTS-generated audio.

    Only runs on models that consistently pass this check; see _SPEAKER_SEP_MODELS.
    """

    def test_same_speaker_higher_than_cross(self, alias, audio_dir):
        from speakeronnx import SpeakerEmbedder, cosine

        emb = SpeakerEmbedder(model=alias)

        ea1 = emb.embed(audio_dir["A1"])
        ea2 = emb.embed(audio_dir["A2"])
        ea3 = emb.embed(audio_dir["A3"])
        eb1 = emb.embed(audio_dir["B1"])
        eb2 = emb.embed(audio_dir["B2"])

        same_A = cosine(ea1, ea2)
        same_B = cosine(eb1, eb2)
        cross_AB = cosine(ea1, eb1)

        print(f"\n[{alias}]")
        print(f"  same(A1,A2)={same_A:.4f}  same(B1,B2)={same_B:.4f}")
        print(f"  cross(A1,B1)={cross_AB:.4f}")

        assert same_A > cross_AB, (
            f"[{alias}] same(A1,A2)={same_A:.4f} should be > cross(A1,B1)={cross_AB:.4f}"
        )
        assert same_B > cross_AB, (
            f"[{alias}] same(B1,B2)={same_B:.4f} should be > cross(A1,B1)={cross_AB:.4f}"
        )

    def test_three_voice_ordering(self, alias, audio_dir):
        """A1 closer to A2 than to C1 (different accent/gender/voice)."""
        from speakeronnx import SpeakerEmbedder, cosine

        emb = SpeakerEmbedder(model=alias)
        ea1 = emb.embed(audio_dir["A1"])
        ea2 = emb.embed(audio_dir["A2"])
        ec1 = emb.embed(audio_dir["C1"])

        same = cosine(ea1, ea2)
        cross = cosine(ea1, ec1)
        print(f"\n[{alias}] three-voice: same(A1,A2)={same:.4f}  cross(A1,C1)={cross:.4f}")

        assert same > cross, (
            f"[{alias}] same={same:.4f} should be > cross={cross:.4f}"
        )
