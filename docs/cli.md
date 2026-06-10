# CLI reference

## `speakeronnx list`

List all available model aliases with metadata.

```bash
speakeronnx list
```

Example output:

```
wespeaker-resnet34
  HF repo  : Wespeaker/wespeaker-voxceleb-resnet34-LM
  File     : voxceleb_resnet34_LM.onnx
  License  : cc-by-4.0
  Embed dim: 256
  Samplerate: 16000 Hz
  WeSpeaker ResNet34 r-vector, large-margin ...

campplus
  HF repo  : csukuangfj/speaker-embedding-models
  File     : wespeaker_en_voxceleb_CAM++_LM.onnx
  License  : cc-by-4.0
  Embed dim: 512
  Samplerate: 16000 Hz
  ...
```

## `speakeronnx embed`

Extract a speaker embedding from a WAV file.

```bash
speakeronnx embed clip.wav
speakeronnx embed clip.wav --model wespeaker-ecapa512
speakeronnx embed clip.wav --model titanet-small --verbose
```

Output:

```
dim=192  norm=1.000000
```

With `--verbose`, the full embedding vector is printed.

## `speakeronnx verify`

Compare two WAV files for same-speaker verification.

```bash
speakeronnx verify a.wav b.wav
speakeronnx verify a.wav b.wav --threshold 0.5
speakeronnx verify a.wav b.wav --model campplus
```

Output:

```
score=0.823456  threshold=0.450000  verdict=SAME
```

Exit code: `0` for SAME, `1` for DIFFERENT. Useful in shell scripts.

## Examples

```bash
# Verify all clips in a directory against an enrollment
for clip in ./test_clips/*.wav; do
    if speakeronnx verify enrollment.wav "$clip"; then
        echo "MATCH: $clip"
    fi
done

# Extract embeddings for enrollment
speakeronnx embed alice_enrollment.wav --model redimnet-b2 > alice_emb.json
```
