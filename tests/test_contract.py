"""Family contract: every Hub download is pinned, and resampler errors propagate."""
import re
import sys
import types

import numpy as np
import pytest

from speakeronnx import embedder
from speakeronnx.embedder import MODEL_REGISTRY

SHA = re.compile(r"[0-9a-f]{40}")


def test_every_registry_entry_pins_a_commit_sha():
    assert MODEL_REGISTRY
    for alias, entry in MODEL_REGISTRY.items():
        assert SHA.fullmatch(entry.revision or ""), f"{alias}: {entry.revision!r}"


def test_download_passes_the_pinned_revision(monkeypatch):
    seen = []
    monkeypatch.setattr(embedder, "hf_hub_download", lambda **kw: seen.append(kw) or "/fake")
    for entry in MODEL_REGISTRY.values():
        entry.download()
    assert len(seen) == len(MODEL_REGISTRY)
    for kw in seen:
        assert SHA.fullmatch(kw["revision"]), kw


def test_resampler_runtime_error_propagates(monkeypatch):
    soxr = types.ModuleType("soxr")

    def broken(*a, **k):
        raise RuntimeError("soxr broke")

    soxr.resample = broken
    monkeypatch.setitem(sys.modules, "soxr", soxr)
    with pytest.raises(RuntimeError, match="soxr broke"):
        embedder.load_audio(np.zeros(800, dtype=np.float32), target_sr=16000, sample_rate=8000)
