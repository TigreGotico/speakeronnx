"""GPU inference example (requires onnxruntime-gpu).

Usage:
    python examples/gpu_inference.py clip.wav
"""
import sys
import numpy as np
from speakeronnx import SpeakerEmbedder


def main():
    if len(sys.argv) < 2:
        print("Usage: python gpu_inference.py <wav_file>")
        sys.exit(1)

    wav_path = sys.argv[1]

    # Try CUDA, fall back to CPU
    providers = [
        "CUDAExecutionProvider",
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]

    embedder = SpeakerEmbedder(
        model="wespeaker-resnet34",
        providers=providers,
    )

    embedding = embedder.embed(wav_path)
    print(f"Embedding dim : {len(embedding)}")
    print(f"L2 norm       : {np.linalg.norm(embedding):.6f}")
    print(f"Provider used : {embedder._session.get_providers()}")


if __name__ == "__main__":
    main()
