"""DIAGNOSTIC ONLY (diag branch): how often does the Windows HNSW flake hit a
filtered drawer query, with and without a windows build, and what does the
fallback return when it does?"""

import json
import logging
import os
import tempfile

import pytest

import mempalace.subwindows as subwindows
from tests.test_subwindows import _long_text, window_palace  # noqa: F401

LOG = os.path.join(tempfile.gettempdir(), "mempalace_diag_flake2.jsonl")


def _setup(col):
    col.update(ids=["long_1"], metadatas=[{"room": "target"}])
    col.upsert(
        ids=[f"d{i}" for i in range(5)],
        documents=[f"migration apples pears {i}" for i in range(5)],
        metadatas=[{"room": "target"}] * 5,
    )
    col.upsert(
        ids=["other-room"],
        documents=[_long_text("zebra quokka migration " * 12, filler_words=16)],
        metadatas=[{"room": "elsewhere"}],
    )


@pytest.mark.parametrize("variant", ["plain", "build"])
@pytest.mark.parametrize("i", range(60))
def test_diag_flake(window_palace, variant, i, monkeypatch, caplog):  # noqa: F811
    import mempalace.searcher as searcher
    from mempalace.searcher import search_memories

    palace, col = window_palace
    _setup(col)
    if variant == "build":
        subwindows.build_windows(palace)
    monkeypatch.setattr(subwindows, "WINDOW_MAX_FETCH", 1)
    notes = {}
    of = searcher._query_drawers_with_filter_fallback

    def spy(*a, **k):
        r = of(*a, **k)
        notes["drawer_rows"] = r["ids"][0]
        return r

    monkeypatch.setattr(searcher, "_query_drawers_with_filter_fallback", spy)
    ow = subwindows.window_candidates

    def spyw(*a, **k):
        r = ow(*a, **k)
        notes["window_best"] = list(r[0])
        return r

    monkeypatch.setattr(subwindows, "window_candidates", spyw)
    caplog.set_level(logging.WARNING)
    r = search_memories("zebra quokka migration", palace, room="target", n_results=1)
    warned = [rec.getMessage()[:90] for rec in caplog.records if "failed" in rec.getMessage()]
    row = {
        "variant": variant,
        "warned": warned,
        "results": [x["drawer_id"] for x in r.get("results", [])],
        **notes,
    }
    with open(LOG, "a") as fh:
        fh.write(json.dumps(row) + "\n")


def test_diag_zz_summary():
    from collections import Counter

    rows = [json.loads(line) for line in open(LOG)] if os.path.exists(LOG) else []
    counts = Counter((r["variant"], bool(r["warned"]), r["results"] == ["long_1"]) for r in rows)
    flaked = [r for r in rows if r["warned"] or r["results"] != ["long_1"]]
    pytest.fail(f"DIAG2 counts={dict(counts)} flaked={flaked}")
