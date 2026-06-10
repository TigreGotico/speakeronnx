"""Batch speaker enrollment and verification.

Enroll multiple speakers from enrollment directories, then verify
unknown clips against the enrolled gallery.

Directory structure:
    enrollments/
        alice/
            clip1.wav
            clip2.wav
        bob/
            clip1.wav
        ...

Usage:
    python examples/batch_enrollment.py enrollments/ test_clip.wav
"""
import os
import sys
import numpy as np
from speakeronnx import SpeakerEmbedder, cosine


def enroll_speakers(enroll_dir: str, embedder: SpeakerEmbedder):
    speaker_embs = {}

    for speaker in sorted(os.listdir(enroll_dir)):
        spk_dir = os.path.join(enroll_dir, speaker)
        if not os.path.isdir(spk_dir):
            continue

        embeddings = []
        for fname in sorted(os.listdir(spk_dir)):
            if not fname.endswith(".wav"):
                continue
            path = os.path.join(spk_dir, fname)
            emb = embedder.embed(path)
            embeddings.append(emb)
            print(f"  Enrolled {speaker}/{fname}  dim={len(emb)}")

        if embeddings:
            # Average enrollment embeddings (L2-norm the mean)
            mean_emb = np.mean(embeddings, axis=0)
            mean_emb /= np.linalg.norm(mean_emb)
            speaker_embs[speaker] = mean_emb
            print(f"  → {speaker}: {len(embeddings)} clips, mean emb shape={mean_emb.shape}")

    return speaker_embs


def main():
    if len(sys.argv) < 3:
        print("Usage: python batch_enrollment.py <enroll_dir/> <test_clip.wav> [model_alias]")
        sys.exit(1)

    enroll_dir = sys.argv[1]
    test_wav = sys.argv[2]
    model_alias = sys.argv[3] if len(sys.argv) > 3 else "wespeaker-resnet34"

    embedder = SpeakerEmbedder(model=model_alias)

    print(f"Enrolling speakers from {enroll_dir}/ ...")
    gallery = enroll_speakers(enroll_dir, embedder)

    print(f"\nVerifying {test_wav} against gallery ...")
    test_emb = embedder.embed(test_wav)

    results = []
    for speaker, enroll_emb in gallery.items():
        score = cosine(test_emb, enroll_emb)
        results.append((score, speaker))

    results.sort(key=lambda x: -x[0])
    print(f"\n{'Speaker':<20} {'Score':>10}")
    print("-" * 32)
    for score, speaker in results:
        print(f"{speaker:<20} {score:>10.6f}")

    best = results[0]
    print(f"\nBest match: {best[1]} (score={best[0]:.4f})")


if __name__ == "__main__":
    main()
