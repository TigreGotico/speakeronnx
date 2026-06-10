"""Speaker verification with threshold tuning.

Demonstrates how to verify whether two audio files are from the same
speaker and tune the decision threshold.

Usage:
    python examples/verify_speakers.py <enroll.wav> <test.wav>
"""
import sys
import numpy as np
from speakeronnx import SpeakerEmbedder, cosine, verify


def main():
    if len(sys.argv) < 3:
        print("Usage: python verify_speakers.py <enroll.wav> <test.wav> [model_alias]")
        sys.exit(1)

    enroll_wav = sys.argv[1]
    test_wav = sys.argv[2]
    model_alias = sys.argv[3] if len(sys.argv) > 3 else "wespeaker-resnet34"

    embedder = SpeakerEmbedder(model=model_alias)

    enroll_emb = embedder.embed(enroll_wav)
    test_emb = embedder.embed(test_wav)

    score = cosine(enroll_emb, test_emb)
    print(f"Model  : {model_alias}")
    print(f"Score  : {score:.6f}")

    # Try multiple thresholds
    for threshold in [0.3, 0.4, 0.45, 0.5, 0.6]:
        ok, _ = verify(enroll_emb, test_emb, threshold=threshold)
        print(f"  threshold={threshold:.2f} → {'SAME' if ok else 'DIFFERENT'}")

    # Recommendation
    if score > 0.5:
        print("\nLikely SAME speaker (score > 0.5)")
    elif score < 0.3:
        print("\nLikely DIFFERENT speaker (score < 0.3)")
    else:
        print("\nAmbiguous — tune threshold on held-out data")


if __name__ == "__main__":
    main()
