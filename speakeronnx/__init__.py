"""speakeronnx — pure-onnxruntime speaker embedding library.

Runtime dependencies: onnxruntime, numpy, huggingface_hub (no torch).

Model registry
--------------
Registered model aliases and their HuggingFace sources:

``"wespeaker-resnet34"`` (default):
    WeSpeaker ResNet34 r-vector trained on VoxCeleb2 Dev (5994 speakers).
    Source: Wespeaker/wespeaker-voxceleb-resnet34-LM (cc-by-4.0).
    Embedding dim: 256. Feature frontend: 80-dim log-Fbank, CMN.

``"wespeaker-ecapa512"``
    WeSpeaker ECAPA-TDNN-512 x-vector trained on VoxCeleb2 Dev.
    Source: Wespeaker/wespeaker-ecapa-tdnn512-LM (cc-by-4.0).
    Embedding dim: 192. Feature frontend: 80-dim log-Fbank, CMN.

Usage
-----
::

    from speakeronnx import SpeakerEmbedder, cosine, verify

    embedder = SpeakerEmbedder(model="wespeaker-resnet34")
    a = embedder.embed("alice1.wav")
    b = embedder.embed("alice2.wav")
    c = embedder.embed("bob.wav")
    print(cosine(a, b))   # > 0.7 same speaker
    print(cosine(a, c))   # < 0.4 different speaker
    ok, score = verify(a, b, threshold=0.45)
"""

from speakeronnx.embedder import (
    SpeakerEmbedder,
    MODEL_REGISTRY,
    ModelEntry,
    cosine,
    verify,
)

__all__ = [
    "SpeakerEmbedder",
    "MODEL_REGISTRY",
    "ModelEntry",
    "cosine",
    "verify",
]
