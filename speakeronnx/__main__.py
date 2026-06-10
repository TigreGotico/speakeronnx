"""CLI entry point: ``speakeronnx``."""
import argparse
import sys


def _cmd_embed(args: argparse.Namespace) -> None:
    from speakeronnx import SpeakerEmbedder
    import numpy as np

    embedder = SpeakerEmbedder(model=args.model)
    emb = embedder.embed(args.wav)
    print(f"dim={emb.shape[0]}  norm={float(np.linalg.norm(emb)):.6f}")
    if args.verbose:
        print(emb.tolist())


def _cmd_verify(args: argparse.Namespace) -> None:
    from speakeronnx import SpeakerEmbedder

    embedder = SpeakerEmbedder(model=args.model)
    a = embedder.embed(args.wav_a)
    b = embedder.embed(args.wav_b)
    accepted, score = embedder.verify(a, b, threshold=args.threshold)
    verdict = "SAME" if accepted else "DIFFERENT"
    print(f"score={score:.6f}  threshold={args.threshold}  verdict={verdict}")
    sys.exit(0 if accepted else 1)


def _cmd_list(args: argparse.Namespace) -> None:
    from speakeronnx import MODEL_REGISTRY

    for alias, entry in MODEL_REGISTRY.items():
        print(f"{alias}")
        print(f"  HF repo  : {entry.hf_repo}")
        print(f"  File     : {entry.hf_file}")
        print(f"  License  : {entry.license}")
        print(f"  Embed dim: {entry.embed_dim}")
        print(f"  Samplerate: {entry.sample_rate} Hz")
        print(f"  {entry.description}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="speakeronnx",
        description="Pure-onnxruntime speaker embedding CLI",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # embed
    p_embed = sub.add_parser("embed", help="Extract embedding from a WAV file")
    p_embed.add_argument("wav", help="Path to WAV file")
    p_embed.add_argument("--model", default="wespeaker-resnet34")
    p_embed.add_argument("--verbose", action="store_true",
                         help="Print full embedding vector")

    # verify
    p_verify = sub.add_parser("verify", help="Compare two WAV files")
    p_verify.add_argument("wav_a")
    p_verify.add_argument("wav_b")
    p_verify.add_argument("--model", default="wespeaker-resnet34")
    p_verify.add_argument("--threshold", type=float, default=0.45)

    # list
    sub.add_parser("list", help="List available models")

    args = parser.parse_args()
    dispatch = {"embed": _cmd_embed, "verify": _cmd_verify, "list": _cmd_list}
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
