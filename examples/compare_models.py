"""Compare multiple models on the same audio files.

Shows how different models perform on the same pair of utterances.
Useful for selecting the right model for your data.

Usage:
    python examples/compare_models.py <same1.wav> <same2.wav> <diff1.wav>
"""
import sys
import numpy as np
from speakeronnx import SpeakerEmbedder, cosine, MODEL_REGISTRY

# Skip models that take raw audio (need different handling)
SKIP_RAW = True


def main():
    if len(sys.argv) < 4:
        print("Usage: python compare_models.py <same1.wav> <same2.wav> <diff1.wav>")
        sys.exit(1)

    same1, same2, diff = sys.argv[1], sys.argv[2], sys.argv[3]

    print(f"{'Model':<22} {'Same-spk':>10} {'Diff-spk':>10} {'Margin':>10}  {'Dim':>4}")
    print("-" * 60)

    results = []
    for alias, entry in MODEL_REGISTRY.items():
        if SKIP_RAW and entry.frontend == "raw":
            continue

        emb = SpeakerEmbedder(model=alias)

        e_same1 = emb.embed(same1)
        e_same2 = emb.embed(same2)
        e_diff = emb.embed(diff)

        same_score = cosine(e_same1, e_same2)
        diff_score = cosine(e_same1, e_diff)
        margin = same_score - diff_score

        results.append((alias, same_score, diff_score, margin, entry.embed_dim))

    results.sort(key=lambda r: -r[3])  # sort by margin descending
    for alias, same, diff, margin, dim in results:
        print(f"{alias:<22} {same:>10.4f} {diff:>10.4f} {margin:>10.4f}  {dim:>4}")

    print("\nModels sorted by same-vs-different margin (higher = better separation)")


if __name__ == "__main__":
    main()
