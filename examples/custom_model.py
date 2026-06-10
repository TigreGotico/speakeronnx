"""Load a custom ONNX model from disk.

Usage:
    python examples/custom_model.py /path/to/model.onnx clip.wav
"""
import sys
import numpy as np
from speakeronnx import SpeakerEmbedder, cosine


def main():
    if len(sys.argv) < 3:
        print("Usage: python custom_model.py <model.onnx> <wav_file> [wav_file2]")
        sys.exit(1)

    onnx_path = sys.argv[1]
    wav1_path = sys.argv[2]

    embedder = SpeakerEmbedder(model=onnx_path)
    print(f"Model        : {onnx_path}")
    print(f"Sample rate  : {embedder.sample_rate} Hz")
    print(f"Embed dim    : {embedder.embed_dim}")
    print(f"Is registered: {embedder.entry is not None}")

    emb1 = embedder.embed(wav1_path)
    print(f"Embedding    : dim={len(emb1)}  norm={np.linalg.norm(emb1):.6f}")

    if len(sys.argv) >= 4:
        wav2_path = sys.argv[3]
        emb2 = embedder.embed(wav2_path)
        score = cosine(emb1, emb2)
        print(f"Cosine({wav1_path}, {wav2_path}): {score:.6f}")


if __name__ == "__main__":
    main()
