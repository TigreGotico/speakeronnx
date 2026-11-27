"""The model tables and counts in the docs match MODEL_REGISTRY."""
import re
from pathlib import Path

import pytest

from speakeronnx import MODEL_REGISTRY

ROOT = Path(__file__).resolve().parent.parent
PAGES = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]


def tables(text):
    """Yield (header, row) cell lists for each table row that names an alias."""
    header = None
    for line in text.splitlines():
        if not line.startswith("|"):
            header = None
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if header is None:
            header = [c.lower() for c in cells]
        elif not set(line) <= set("|-: "):
            yield header, cells


def rows():
    for page in PAGES:
        for header, cells in tables(page.read_text(encoding="utf-8")):
            alias = re.fullmatch(r"`([a-z0-9-]+)`", cells[0])
            if alias and alias.group(1) in MODEL_REGISTRY:
                yield pytest.param(header, cells, MODEL_REGISTRY[alias.group(1)],
                                   id=f"{page.name}:{alias.group(1)}")


def stated_params(entry):
    found = re.search(r"([\d.]+)M params", entry.description)
    return f"{float(found.group(1)):.1f}M" if found else "not stated"


@pytest.mark.parametrize("header, cells, entry", list(rows()))
def test_a_model_row_matches_its_registry_entry(header, cells, entry):
    for name, cell in zip(header, cells):
        if "dim" in name:
            assert cell == str(entry.embed_dim)
        elif name == "frontend":
            assert cell == entry.frontend
        elif name == "license":
            assert cell == entry.license
        elif name == "params":
            assert cell == stated_params(entry)
        elif name == "hf repo":
            assert cell.startswith(f"[{entry.hf_repo}]")


def aliases_in_table(page, header_cell):
    """The aliases of the one table whose header has header_cell, by row."""
    found, inside = [], False
    for line in (ROOT / page).read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            if inside:
                break
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not inside:
            inside = header_cell in [c.lower() for c in cells]
            continue
        found += re.findall(r"`([a-z0-9-]+)`", cells[0])
    return found


@pytest.mark.parametrize("page, header_cell", [
    ("README.md", "frontend"),
    ("README.md", "hf repo"),
    ("docs/index.md", "params"),
])
def test_a_full_model_table_lists_every_registered_model_once(page, header_cell):
    found = aliases_in_table(page, header_cell)
    assert sorted(found) == sorted(MODEL_REGISTRY)


def test_the_model_guide_lists_only_registered_models():
    text = (ROOT / "docs" / "models.md").read_text(encoding="utf-8")
    listed = {m for m in re.findall(r"^\| `([a-z0-9-]+)` \|", text, re.M)}
    assert listed and listed <= set(MODEL_REGISTRY)


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_a_stated_model_count_is_the_registry_size(page):
    for count in re.findall(r"registers (\d+) models", page.read_text(encoding="utf-8")):
        assert int(count) == len(MODEL_REGISTRY)
