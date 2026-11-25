"""The daemon's mine child: compute at idle priority, every write in the daemon.

These run real child processes. The child gets a deterministic embedder (the
same shape as ``conftest``'s) through a small launcher, so no ONNX model is
loaded and the run stays fast.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
import time

import pytest
from conftest import _TEST_EMBED_DIM

from mempalace import daemon
from mempalace.mine_supervisor import run_mine_in_child

_EMBEDDER = f"""
import hashlib, math, re
_DIM = {_TEST_EMBED_DIM}
_TOK = re.compile(r"[a-z0-9_]+")

def _vec(text):
    vec = [0.0] * _DIM
    for token in _TOK.findall((text or "").lower()) or [""]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        vec[int.from_bytes(digest[:4], "little") % _DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]

import mempalace.backends.embedding_wrapper as _ew
_ew._embed_texts = lambda texts, **_: [_vec(str(t)) for t in texts]
_ew._resolve_document_embedder = lambda: None
"""

# The real child, with the test embedder in place of the configured one.
REAL_CHILD = (
    _EMBEDDER + "\nimport sys\nfrom mempalace import mine_child\nsys.exit(mine_child.main())\n"
)

# A child that speaks the protocol, opens the drawers collection, then stalls
# until RELEASE exists -- a mine stuck in its compute.
STALLING_CHILD = textwrap.dedent(
    """
    import os, sys, time
    from mempalace.mine_child import _read_msg, _write_msg
    rx, tx = sys.stdin.buffer, sys.stdout.buffer
    job = _read_msg(rx)
    _write_msg(tx, {"op": "started", "pid": os.getpid(), "priority": "test"})
    _write_msg(tx, {"op": "open", "kwargs": {"collection_name": None, "create": True}})
    opened = _read_msg(rx)
    assert opened["ok"], opened
    release = os.environ["TEST_RELEASE"]
    while not os.path.exists(release):
        time.sleep(0.05)
    _write_msg(tx, {"op": "call", "handle": opened["value"]["handle"], "method": "count",
                    "args": (), "kwargs": {}})
    counted = _read_msg(rx)
    _write_msg(tx, {"op": "done", "result": {"success": True, "count": counted["value"]}})
    """
)


def _child_argv(script: str) -> list[str]:
    return [sys.executable, "-c", script]


def _project(tmp_path):
    src = tmp_path / "proj"
    src.mkdir()
    (src / "notes.md").write_text(
        "# Field notes\n\n"
        + "The heron waited by the reed bed until the tide turned silver.\n" * 20,
        encoding="utf-8",
    )
    return src


def _counting_daemon_embedder(monkeypatch):
    """Record every text the daemon process itself embeds."""
    from conftest import _StableTestEmbeddingFunction

    import mempalace.backends.chroma as chroma_mod

    seen: list[str] = []
    base = _StableTestEmbeddingFunction()

    class Counting(_StableTestEmbeddingFunction):
        def __call__(self, input):
            seen.extend(str(t) for t in list(input or []))
            return base(input=input)

    ef = Counting()
    monkeypatch.setattr(
        chroma_mod.ChromaBackend, "_resolve_embedding_function", staticmethod(lambda **_: ef)
    )
    return seen


def test_mine_runs_in_child_and_daemon_stores_its_vectors(tmp_path, monkeypatch):
    """A whole project mine through the child: the drawers land verbatim, and
    the daemon only writes -- it embeds none of the mined text."""
    palace = tmp_path / "palace"
    src = _project(tmp_path)
    seen = _counting_daemon_embedder(monkeypatch)

    result = run_mine_in_child(
        {"palace_path": str(palace), "source": str(src), "mode": "projects", "wing": "proj"},
        palace_path=str(palace),
        write_lock=threading.RLock(),
        argv=_child_argv(REAL_CHILD),
    )

    assert result["success"] is True, result
    assert result["child"]["calls"] > 0
    assert result["child"]["pid"] != os.getpid()
    assert not any("heron" in text for text in seen), "the daemon re-embedded mined text"

    from mempalace.palace import get_collection

    stored = get_collection(str(palace)).get(where={"wing": "proj"}, include=["documents"])
    joined = "\n".join(stored.documents)
    assert "The heron waited by the reed bed until the tide turned silver." in joined


@pytest.mark.skipif(not hasattr(os, "SCHED_IDLE"), reason="SCHED_IDLE is Linux-only")
def test_child_drops_to_sched_idle_before_heavy_imports():
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, sys\n"
            "from mempalace.mine_child import lower_priority\n"
            "applied = lower_priority()\n"
            "heavy = [m for m in ('onnxruntime', 'chromadb', 'numpy') if m in sys.modules]\n"
            "print(applied, os.sched_getscheduler(0) == os.SCHED_IDLE, heavy)\n",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert out.stdout.split()[:2] == ["sched-idle", "True"]
    assert out.stdout.strip().endswith("[]")


def test_backend_open_inside_child_is_refused():
    """Every storage path in the child goes to the daemon; a direct backend open
    fails loudly instead of quietly becoming a second Chroma writer."""
    script = textwrap.dedent(
        """
        from mempalace import mine_child, miner, convo_miner, palace
        mine_child._install_remote_storage()
        assert miner.get_collection.__module__ == "mempalace.mine_child"
        assert convo_miner.get_collection.__module__ == "mempalace.mine_child"
        assert palace.get_collection.__module__ == "mempalace.mine_child"
        from mempalace.backends.chroma import ChromaBackend
        try:
            ChromaBackend().get_collection("/nonexistent", collection_name="x", create=True)
        except mine_child.ChildWriterViolation:
            print("refused")
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "refused"


def _wait_for(predicate, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("condition not met in time")


def test_diary_write_lands_while_a_mine_runs(tmp_path, monkeypatch):
    """A Stop-hook checkpoint is not held behind a long mine: the diary lane
    files it while the mine child is still working."""
    monkeypatch.setenv(daemon.STATE_ROOT_ENV, str(tmp_path / "state"))
    monkeypatch.setenv("MEMPALACE_DAEMON_MINE_CHILD", "1")
    release = tmp_path / "release"
    monkeypatch.setenv("TEST_RELEASE", str(release))
    palace = tmp_path / "palace"
    palace.mkdir()

    runtime = daemon.DaemonRuntime(str(palace))
    runtime.mine_child_argv = _child_argv(STALLING_CHILD)
    runtime.start_worker()
    try:
        mine = runtime.store.enqueue("mine", {"source": str(tmp_path)}, dedupe_key="m")
        runtime.worker_wake.set()
        _wait_for(lambda: runtime.active_child is not None and runtime.active_child.calls >= 1)

        diary = runtime.store.enqueue(
            "diary_write",
            {"agent_name": "tester", "entry": "checkpoint filed mid-mine", "topic": "t"},
            priority=10,
        )
        runtime.worker_wake.set()
        finished = _wait_for(
            lambda: (
                runtime.store.get(diary.id)
                if runtime.store.get(diary.id).state in daemon.TERMINAL_STATES
                else None
            )
        )
        assert finished.state == "succeeded", finished.error
        # The mine is still running: the checkpoint did not wait for it.
        assert runtime.store.get(mine.id).state == "running"
        assert runtime.progress()["pid"] == runtime.active_child.pid

        release.write_text("go")
        done = _wait_for(
            lambda: (
                runtime.store.get(mine.id)
                if runtime.store.get(mine.id).state in daemon.TERMINAL_STATES
                else None
            )
        )
        assert done.state == "succeeded", done.error
        assert done.result["count"] >= 1  # the diary drawer, seen through the daemon
    finally:
        release.write_text("go")
        runtime.shutdown_event.set()
        for thread in runtime.worker_threads.values():
            thread.join(timeout=30)


def test_child_that_dies_fails_the_job_and_keeps_the_daemon(tmp_path):
    palace = tmp_path / "palace"
    result = run_mine_in_child(
        {"palace_path": str(palace)},
        palace_path=str(palace),
        write_lock=threading.RLock(),
        argv=_child_argv("import sys; sys.stdin.buffer.read(4); sys.exit(3)"),
    )
    assert result["success"] is False
    assert result["error_class"] == "MineChildDied"
    assert "code 3" in result["error"]


def test_mine_child_env_caps_threads_and_turns_off_spinning(monkeypatch):
    from mempalace.embedding import _intra_op_session_options
    from mempalace.mine_supervisor import _child_env

    monkeypatch.delenv("MEMPALACE_EMBEDDING_THREADS", raising=False)
    monkeypatch.delenv("MEMPALACE_ORT_ALLOW_SPINNING", raising=False)
    env = _child_env("/palace", 2)
    assert env["MEMPALACE_EMBEDDING_THREADS"] == "2"
    assert env["MEMPALACE_ORT_ALLOW_SPINNING"] == "0"

    # A user's own thread setting wins over the daemon default.
    monkeypatch.setenv("MEMPALACE_EMBEDDING_THREADS", "6")
    assert _child_env("/palace", 2)["MEMPALACE_EMBEDDING_THREADS"] == "6"

    monkeypatch.setenv("MEMPALACE_ORT_ALLOW_SPINNING", "0")
    so = _intra_op_session_options(0)
    assert so is not None
    assert so.get_session_config_entry("session.intra_op.allow_spinning") == "0"
    assert _intra_op_session_options(2).intra_op_num_threads == 2
