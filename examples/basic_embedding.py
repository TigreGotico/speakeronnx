"""Basic speaker embedding extraction.

Usage:
    python examples/basic_embedding.py path/to/speaker.wav
"""
import sys
import numpy as np
from speakeronnx import SpeakerEmbedder, cosine


def main():
    if len(sys.argv) < 2:
        print("Usage: python basic_embedding.py <wav_file> [model_alias]")
        sys.exit(1)

    wav_path = sys.argv[1]
    model_alias = sys.argv[2] if len(sys.argv) > 2 else "wespeaker-resnet34"

    embedder = SpeakerEmbedder(model=model_alias)
    embedding = embedder.embed(wav_path)

    print(f"Model : {model_alias}")
    print(f"Dim   : {len(embedding)}")
    print(f"Norm  : {np.linalg.norm(embedding):.6f}")
    print(f"First 8 values: {embedding[:8].tolist()}")
    print(f"Self-cosine    : {cosine(embedding, embedding):.6f}  (should be 1.0)")


if __name__ == "__main__":
    main()
