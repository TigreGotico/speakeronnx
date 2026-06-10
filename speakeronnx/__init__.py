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

``"wespeaker-resnet293"``
    WeSpeaker ResNet293 r-vector (large-margin finetuned).
    Source: Wespeaker/wespeaker-voxceleb-resnet293-LM (cc-by-4.0).
    Embedding dim: 256. 28.62M params.

``"campplus"``
    WeSpeaker CAM++ (large-margin finetuned) on VoxCeleb2 Dev.
    Source: csukuangfj/speaker-embedding-models (cc-by-4.0).
    Embedding dim: 512. D-TDNN backbone.

``"campplus-zh-en"``
    3D-Speaker CAM++ advanced, multilingual (zh+en).
    Source: csukuangfj/speaker-embedding-models (apache-2.0).
    Embedding dim: 192.

``"eres2net"``
    3D-Speaker ERes2Net trained on VoxCeleb.
    Source: csukuangfj/speaker-embedding-models (apache-2.0).
    Embedding dim: 192.

``"titanet-small"``
    NVIDIA NeMo TitaNet-small.
    Source: csukuangfj/speaker-embedding-models (cc-by-4.0).
    Embedding dim: 192. Input: [B, 80, T] transposed fbank.

``"titanet-large"``
    NVIDIA NeMo TitaNet-large.
    Source: csukuangfj/speaker-embedding-models (cc-by-4.0).
    Embedding dim: 192. Larger capacity.

``"redimnet-b2"``
    ReDimNet b2 (Reshape Dimensions Network, Interspeech 2024).
    Source: OpenVoiceOS/redimnet-b2-vox2-onnx (apache-2.0).
    Embedding dim: 192. Compact 1.8M params. Accepts raw audio
    with internal frontend (72 mel bins, f_max=7600 Hz).

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
    DEFAULT_MODEL,
    cosine,
    verify,
)

__all__ = [
    "SpeakerEmbedder",
    "MODEL_REGISTRY",
    "ModelEntry",
    "DEFAULT_MODEL",
    "cosine",
    "verify",
]
