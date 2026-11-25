"""Daemon side of a mine run in a supervised child (see ``mine_child``).

The daemon owns the palace: it holds the writer lock and the only storage
client. :func:`run_mine_in_child` starts ``mempalace.mine_child`` at idle
priority, hands it the job, and serves its storage calls one at a time under
``write_lock`` -- the lock every other daemon write (a diary entry) also takes,
so those land between the mine's calls rather than after the whole mine.

Every message from the child is a heartbeat (``MineChild.last_progress``); the
child can be killed from another thread (``MineChild.kill``) without taking
the daemon down, which an in-process mine cannot offer (#2917).
"""

from __future__ import annotations

import os
import pickle
import subprocess
import threading
import time
from typing import Any, Callable, Optional

from ._proc import safe_child_env, safe_module_argv
from .mine_child import _read_msg, _write_msg

# Cap on the child's captured stdout/stderr kept for the job result.
_OUTPUT_TAIL_BYTES = 64 * 1024
# Rows per page when the child streams a metadata scan.
_ITER_PAGE = 2000


def _child_env(palace_path: str, threads: int) -> dict:
    env = safe_child_env()
    env["MEMPALACE_PALACE_PATH"] = palace_path
    # Background mining: a couple of intra-op threads cost far less CPU per
    # row than the default pool (0.108 CPU-s/row at 1 thread, 0.41 at 8 in the
    # #3058 measurements), and spinning threads burn idle cores for nothing.
    if threads > 0 and "MEMPALACE_EMBEDDING_THREADS" not in os.environ:
        env["MEMPALACE_EMBEDDING_THREADS"] = str(threads)
    env.setdefault("MEMPALACE_ORT_ALLOW_SPINNING", "0")
    return env


def _popen_kwargs() -> dict:
    kwargs: dict[str, Any] = {
        "stdin": subprocess.PIPE,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "close_fds": True,
    }
    if os.name == "nt":  # pragma: no cover - Windows
        flags = 0
        for name in ("CREATE_NO_WINDOW", "IDLE_PRIORITY_CLASS"):
            flags |= getattr(subprocess, name, 0)
        kwargs["creationflags"] = flags
    return kwargs


def _picklable_exc(exc: BaseException) -> BaseException:
    try:
        pickle.loads(pickle.dumps(exc))
        return exc
    except Exception:
        return RuntimeError(f"{type(exc).__name__}: {exc}")


class MineChild:
    """One running mine child and the storage handles it opened."""

    def __init__(self, proc: subprocess.Popen):
        self.proc = proc
        self.pid = proc.pid
        self.started = time.monotonic()
        self.last_progress = self.started
        self.calls = 0
        self.priority: Optional[str] = None
        self.done_at: Optional[float] = None
        self.killed_reason: Optional[str] = None
        self._output = bytearray()
        self._output_lock = threading.Lock()
        self._reader = threading.Thread(target=self._drain_stderr, daemon=True)
        self._reader.start()

    def _drain_stderr(self) -> None:
        stream = self.proc.stderr
        while True:
            chunk = stream.read1(8192) if hasattr(stream, "read1") else stream.read(8192)
            if not chunk:
                return
            with self._output_lock:
                self._output.extend(chunk)
                if len(self._output) > _OUTPUT_TAIL_BYTES:
                    del self._output[: len(self._output) - _OUTPUT_TAIL_BYTES]

    def output(self) -> str:
        self._reader.join(timeout=5)
        with self._output_lock:
            return bytes(self._output).decode("utf-8", errors="replace")

    def kill(self, reason: str) -> None:
        """Stop the child. The daemon keeps running; the job is failed."""
        self.killed_reason = reason
        try:
            self.proc.kill()
        except OSError:
            pass

    def progress(self) -> dict:
        now = time.monotonic()
        return {
            "pid": self.pid,
            "calls": self.calls,
            "priority": self.priority,
            "running_seconds": round(now - self.started, 1),
            "seconds_since_progress": round(now - self.last_progress, 1),
        }


def run_mine_in_child(
    payload: dict,
    *,
    palace_path: str,
    write_lock: threading.Lock,
    threads: int = 2,
    on_start: Optional[Callable[[MineChild], None]] = None,
    argv: Optional[list] = None,
) -> dict:
    """Run one mine job in a child process; return the job result.

    ``argv`` overrides the child command (tests use a fake child).
    """
    from .palace import _validate_palace_fts5_after_mine, get_collection, palace_write_serial

    cmd = argv or safe_module_argv("mempalace.mine_child")
    proc = subprocess.Popen(cmd, env=_child_env(palace_path, threads), **_popen_kwargs())
    child = MineChild(proc)
    if on_start is not None:
        on_start(child)
    handles: dict[int, Any] = {}
    gens: dict[int, Any] = {}
    writes_at_start = palace_write_serial()
    result: Optional[dict] = None
    try:
        _write_msg(proc.stdin, {"payload": payload})
        while True:
            try:
                msg = _read_msg(proc.stdout)
            except (EOFError, OSError, pickle.UnpicklingError):
                break
            child.last_progress = time.monotonic()
            op = msg.get("op")
            if op == "started":
                child.priority = msg.get("priority")
                continue
            if op == "progress":
                continue
            if op == "done":
                result = dict(msg.get("result") or {})
                child.done_at = time.monotonic()
                break
            child.calls += 1
            try:
                with write_lock:
                    value = _serve(
                        op,
                        msg,
                        palace_path,
                        handles,
                        gens,
                        writes_at_start,
                        get_collection,
                        _validate_palace_fts5_after_mine,
                    )
                reply = {"ok": True, "value": value}
            except BaseException as exc:  # noqa: BLE001 - every failure goes back to the child
                reply = {"ok": False, "exc": _picklable_exc(exc)}
            try:
                _write_msg(proc.stdin, reply)
            except (BrokenPipeError, OSError):
                break
    finally:
        try:
            proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            # Reported done but did not exit (#2917): its work is in; reap it.
            child.kill("did not exit 30 s after reporting done")
            proc.wait()
    output = child.output()
    if result is None:
        reason = child.killed_reason or f"mine child exited with code {proc.returncode}"
        result = {"success": False, "error": reason, "error_class": "MineChildDied", "exit_code": 1}
    result.setdefault("success", True)
    result.setdefault("exit_code", 0 if result.get("success") else 1)
    result["child"] = child.progress()
    if output:
        result["stdout"] = output
    return result


def _serve(op, msg, palace_path, handles, gens, writes_at_start, get_collection, validate):
    if op == "open":
        kwargs = dict(msg.get("kwargs") or {})
        coll = get_collection(palace_path, **kwargs)
        handle = len(handles) + 1
        handles[handle] = coll
        caps = _caps(coll, palace_path, kwargs.get("backend"))
        return {"handle": handle, "caps": caps}
    if op == "call":
        coll = handles[msg["handle"]]
        method = msg["method"]
        if method.startswith("_"):
            raise AttributeError(method)
        return getattr(coll, method)(*msg.get("args", ()), **msg.get("kwargs", {}))
    if op == "iter":
        coll = handles[msg["handle"]]
        gen_id = len(gens) + 1
        gens[gen_id] = iter(coll.iter_metadata(*msg.get("args", ()), **msg.get("kwargs", {})))
        return gen_id
    if op == "next":
        gen = gens[msg["gen"]]
        items = []
        for item in gen:
            items.append(item)
            if len(items) >= _ITER_PAGE:
                return {"items": items, "done": False}
        del gens[msg["gen"]]
        return {"items": items, "done": True}
    if op == "validate":
        return validate(palace_path, writes_since=writes_at_start)
    raise ValueError(f"unknown mine channel op: {op!r}")


def _caps(coll, palace_path, backend) -> dict:
    from .backends.embedding_wrapper import EmbeddingCollection

    inner = coll._inner if isinstance(coll, EmbeddingCollection) else coll
    server_embedder = False
    try:
        from .palace.backend import get_backend_for_palace

        server_embedder = (
            "server_embedder" in get_backend_for_palace(palace_path, explicit=backend).capabilities
        )
    except Exception:
        pass
    methods = sorted(
        name
        for name in dir(inner)
        if not name.startswith("_") and callable(getattr(type(inner), name, None))
    )
    return {
        "iter_metadata": hasattr(inner, "iter_metadata"),
        "upsert_merges_metadata": bool(getattr(coll, "upsert_merges_metadata", False)),
        "distance_metric": getattr(coll, "distance_metric", "cosine"),
        "server_embedder": server_embedder,
        "methods": methods,
    }
