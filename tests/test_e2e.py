"""End-to-end tests — downloads real ONNX models and generates real audio.

Requires: edge-tts (pip install edge-tts) and ffmpeg on PATH.
Run with: pytest tests/test_e2e.py -v -s

These tests:
1. Generate 3 utterances for voice A (en-US-GuyNeural) and 3 for voice B
   (en-US-JennyNeural) via edge-tts, convert to 16k mono WAV with ffmpeg.
2. For each registered model, assert:
   - same-speaker cosine(A1, A2) > different-speaker cosine(A1, B1)
   - same-speaker cosine(B1, B2) > different-speaker cosine(B1, A1)
   - embed(f) is identical on two calls (determinism)
"""

import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pytest


UTTERANCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Speaker verification systems identify who is talking.",
    "Voice biometrics provide an additional layer of security.",
]

VOICE_A = "en-US-GuyNeural"
VOICE_B = "en-US-JennyNeural"
VOICE_C = "en-GB-RyanNeural"


def _tts_to_wav(text: str, voice: str, path: str, timeout: int = 60) -> None:
    """Generate speech via edge-tts and convert to 16k mono WAV."""
    mp3 = path.replace(".wav", ".mp3")
    subprocess.run(
        ["edge-tts", "--voice", voice, "--text", text, "--write-media", mp3],
        check=True, timeout=timeout, capture_output=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp3, "-ar", "16000", "-ac", "1", "-f", "wav", path],
        check=True, timeout=30, capture_output=True,
    )
    os.unlink(mp3)


def _skip_if_no_tts():
    try:
        subprocess.run(["edge-tts", "--version"], capture_output=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pytest.skip("edge-tts not available")
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pytest.skip("ffmpeg not available")


@pytest.fixture(scope="module")
def audio_dir(tmp_path_factory):
    """Generate all test audio clips once per module."""
    _skip_if_no_tts()
    d = tmp_path_factory.mktemp("audio")

    clips = {}
    for voice, tag in [(VOICE_A, "A"), (VOICE_B, "B"), (VOICE_C, "C")]:
        for i, text in enumerate(UTTERANCES):
            path = str(d / f"{tag}{i+1}.wav")
            _tts_to_wav(text, voice, path)
            clips[f"{tag}{i+1}"] = path

    return clips


@pytest.mark.parametrize("alias", [
    "wespeaker-resnet34",
    "wespeaker-ecapa512",
    "wespeaker-resnet293",
    "campplus",
    "campplus-zh-en",
    "eres2net",
    "titanet-small",
    "titanet-large",
    "redimnet-b2",
])
class TestE2ESpeakerVerification:
    def test_same_speaker_higher_than_cross(self, alias, audio_dir):
        from speakeronnx import SpeakerEmbedder, cosine

        emb = SpeakerEmbedder(model=alias)

        # Embed all clips
        ea1 = emb.embed(audio_dir["A1"])
        ea2 = emb.embed(audio_dir["A2"])
        ea3 = emb.embed(audio_dir["A3"])
        eb1 = emb.embed(audio_dir["B1"])
        eb2 = emb.embed(audio_dir["B2"])

        same_A = cosine(ea1, ea2)
        same_A2 = cosine(ea1, ea3)
        same_B = cosine(eb1, eb2)
        cross_AB = cosine(ea1, eb1)
        cross_AB2 = cosine(ea2, eb2)

        print(f"\n[{alias}]")
        print(f"  same(A1,A2)={same_A:.4f}  same(A1,A3)={same_A2:.4f}  same(B1,B2)={same_B:.4f}")
        print(f"  cross(A1,B1)={cross_AB:.4f}  cross(A2,B2)={cross_AB2:.4f}")

        assert same_A > cross_AB, (
            f"[{alias}] same(A1,A2)={same_A:.4f} should be > cross(A1,B1)={cross_AB:.4f}"
        )
        assert same_B > cross_AB, (
            f"[{alias}] same(B1,B2)={same_B:.4f} should be > cross(A1,B1)={cross_AB:.4f}"
        )

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
