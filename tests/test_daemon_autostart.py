"""Hook-side daemon start: stdlib-only enqueue, detached spawn, one daemon (#3058)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from mempalace import daemon, daemon_queue, hooks_cli
from mempalace.write_routing import ResolvedWriteRoutingPolicy, WriteRoutingPolicy

HEAVY = ("chromadb", "numpy", "onnxruntime", "mempalace.palace", "mempalace.backends")


def _run_py(code: str, env: dict | None = None, timeout: float = 60) -> str:
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, **(env or {})},
        check=True,
    )
    return out.stdout.strip()


def test_queue_module_imports_no_storage_stack():
    # A fresh interpreter: this pytest process already has numpy loaded.
    loaded = _run_py(
        f"""
        import sys, json
        import mempalace.daemon_queue, mempalace.hooks_cli
        print(json.dumps([m for m in {HEAVY!r} if m in sys.modules]))
        """
    )
    assert json.loads(loaded) == []


def test_hook_daemon_enqueue_imports_no_storage_stack_and_fits_the_budget(tmp_path):
    """The whole hook-side daemon route -- routing, probe, enqueue, spawn --
    stays off the storage stack and inside the 500 ms hook budget."""
    palace = tmp_path / "palace"
    palace.mkdir()
    state_root = tmp_path / "state"
    # A queue that already exists, as on every hook after the first.
    with patch.dict(os.environ, {daemon_queue.STATE_ROOT_ENV: str(state_root)}):
        daemon_queue.QueueStore(daemon_queue.queue_path(str(palace)))
    out = _run_py(
        f"""
        import json, sys, time
        t0 = time.perf_counter()
        from mempalace import daemon_queue, hooks_cli
        spawned = []
        daemon_queue.subprocess.Popen = lambda *a, **k: spawned.append(a)
        with hooks_cli._hook_trigger("stop"), hooks_cli._hook_write_routing_context() as r:
            job = hooks_cli._submit_daemon_job(
                "mine", {{"source": "/x", "mode": "convos"}}, dedupe_key="k"
            )
        elapsed = time.perf_counter() - t0
        print(json.dumps({{
            "elapsed": elapsed,
            "use_daemon": r.use_daemon,
            "spawned": len(spawned),
            "heavy": [m for m in {HEAVY!r} if m in sys.modules],
            "job": job,
        }}))
        """,
        env={
            "MEMPALACE_PALACE_PATH": str(palace),
            "MEMPALACE_HOOK_WRITE_ROUTING": "prefer",
            daemon_queue.STATE_ROOT_ENV: str(state_root),
            "HOME": str(tmp_path / "home"),
        },
    )
    res = json.loads(out)
    assert res["heavy"] == []
    assert res["use_daemon"] is True, "prefer + no daemon + auto-start routes to the daemon"
    assert res["spawned"] == 1
    assert res["job"]["state"] == "queued"
    # Generous on CI runners; the real budget everywhere else.
    budget = 2.0 if os.environ.get("CI") else 0.5
    assert res["elapsed"] < budget, res["elapsed"]


class _Cfg:
    palace_path = "/tmp/palace"

    def __init__(self, policy, autostart=True):
        self._policy = policy
        self.hook_daemon_autostart = autostart

    def resolve_write_routing(self, scope):
        return ResolvedWriteRoutingPolicy(policy=self._policy, source="test")


@pytest.mark.parametrize(
    "policy,autostart,alive,expect",
    [
        (WriteRoutingPolicy.PREFER, True, False, ("daemon", True)),
        (WriteRoutingPolicy.REQUIRE, True, False, ("daemon", True)),
        (WriteRoutingPolicy.PREFER, False, False, ("direct", False)),
        (WriteRoutingPolicy.REQUIRE, False, False, ("blocked", False)),
        # Registered and alive but not answering: cannot start a second one.
        (WriteRoutingPolicy.PREFER, True, True, ("direct", False)),
        (WriteRoutingPolicy.DIRECT, True, False, ("direct", False)),
    ],
)
def test_routing_auto_start_keeps_policy_semantics(monkeypatch, policy, autostart, alive, expect):
    monkeypatch.setattr(hooks_cli, "MempalaceConfig", lambda: _Cfg(policy, autostart))
    monkeypatch.setattr(hooks_cli, "_daemon_available", lambda: False)
    monkeypatch.setattr(daemon_queue, "daemon_registered_alive", lambda _p: alive)
    decision = hooks_cli._compute_hook_write_routing().decision
    assert (decision.target.value, decision.auto_start_daemon) == expect


def test_default_policy_never_starts_a_daemon(tmp_path, monkeypatch):
    """Daemon mode stays opt-in: auto-start only acts under prefer/require."""
    from mempalace.config import MempalaceConfig

    for key in (
        "MEMPALACE_HOOK_WRITE_ROUTING",
        "MEMPALACE_WRITE_ROUTING",
        "MEMPALACE_HOOKS_DAEMON",
    ):
        monkeypatch.delenv(key, raising=False)
    cfg = MempalaceConfig(config_dir=str(tmp_path))
    assert cfg.hook_daemon_autostart is True
    monkeypatch.setattr(hooks_cli, "MempalaceConfig", lambda: cfg)
    with patch.object(hooks_cli, "_daemon_available") as probe:
        decision = hooks_cli._compute_hook_write_routing().decision
    assert decision.target.value == "direct"
    probe.assert_not_called()


def test_autostart_flag_reads_env_and_config(tmp_path, monkeypatch):
    from mempalace.config import MempalaceConfig

    (tmp_path / "config.json").write_text(json.dumps({"hooks": {"daemon_autostart": False}}))
    monkeypatch.delenv("MEMPALACE_HOOK_DAEMON_AUTOSTART", raising=False)
    assert MempalaceConfig(config_dir=str(tmp_path)).hook_daemon_autostart is False
    monkeypatch.setenv("MEMPALACE_HOOK_DAEMON_AUTOSTART", "1")
    assert MempalaceConfig(config_dir=str(tmp_path)).hook_daemon_autostart is True


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setenv(daemon_queue.STATE_ROOT_ENV, str(tmp_path / "state"))
    palace = tmp_path / "palace"
    palace.mkdir()
    return str(palace)


def test_spawn_is_detached_and_skipped_while_a_daemon_is_alive(state, monkeypatch):
    with patch.object(daemon_queue.subprocess, "Popen") as popen:
        monkeypatch.setattr(daemon_queue, "daemon_registered_alive", lambda _p: True)
        assert daemon_queue.spawn_daemon_detached(state) is False
        popen.assert_not_called()

        monkeypatch.setattr(daemon_queue, "daemon_registered_alive", lambda _p: False)
        assert daemon_queue.spawn_daemon_detached(state) is True
        assert daemon_queue.spawn_daemon_detached(state) is False, "throttled while it comes up"
    popen.assert_called_once()
    argv, kwargs = popen.call_args.args[0], popen.call_args.kwargs
    assert "serve" in argv and state in argv or daemon_queue.canonical_palace_path(state) in argv
    if os.name != "nt":
        assert kwargs["start_new_session"] is True
    assert kwargs["stdin"] is subprocess.DEVNULL
    # A hook-started daemon exits when idle unless config says otherwise.
    flag = argv.index("--idle-exit-default")
    assert argv[flag + 1] == str(daemon_queue.AUTOSTART_IDLE_EXIT_MINUTES)


def test_second_server_refuses_while_the_singleton_is_held(state):
    with daemon_queue.daemon_singleton(state) as won:
        assert won is True
        with daemon_queue.daemon_singleton(state) as again:
            assert again is False
        with pytest.raises(daemon.DaemonAlreadyRunning):
            daemon.run_server(state)
    with daemon_queue.daemon_singleton(state) as won_after_release:
        assert won_after_release is True


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group cleanup in the test")
def test_racing_detached_spawns_leave_exactly_one_daemon(state, tmp_path):
    """Two hooks spawning at the same moment: one daemon serves, the other exits 0."""
    env = {**os.environ, "MEMPALACE_PALACE_PATH": state}
    argv = [sys.executable, "-m", "mempalace.daemon", "serve", "--palace", state]
    procs = [
        subprocess.Popen(argv, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(2)
    ]
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            exited = [p for p in procs if p.poll() is not None]
            if exited and daemon.get_client_if_running(state, health_timeout=0.5):
                break
            time.sleep(0.1)
        exited = [p for p in procs if p.poll() is not None]
        alive = [p for p in procs if p.poll() is None]
        assert len(exited) == 1 and len(alive) == 1
        assert exited[0].returncode == daemon_queue.EXIT_ALREADY_RUNNING
        assert daemon_queue._registered_pid(state) == alive[0].pid
    finally:
        client = daemon.get_client_if_running(state, health_timeout=0.5)
        if client is not None:
            client.shutdown()
        for p in procs:
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()


def test_daemon_diary_job_writes_the_hook_ack_only_after_filing(tmp_path, monkeypatch):
    from mempalace import service

    monkeypatch.setattr(hooks_cli, "STATE_DIR", tmp_path)
    ack = {"messages_folded": 4, "ack_ts": "t", "session_id": "s/1", "exchange_count": 9}

    import mempalace.mcp_server as mcp

    monkeypatch.setattr(mcp, "tool_diary_write", lambda **kw: {"success": False, "error": "x"})
    service.run_diary_write({"entry": "e", "hook_ack": ack})
    assert not (tmp_path / "last_checkpoint").exists()

    monkeypatch.setattr(mcp, "tool_diary_write", lambda **kw: {"success": True, "entry_id": "d"})
    service.run_diary_write({"entry": "e", "hook_ack": ack})
    assert json.loads((tmp_path / "last_checkpoint").read_text())["msgs"] == 4
    markers = list(Path(tmp_path).glob("*_last_save"))
    assert [m.read_text() for m in markers] == ["9"]


def _serve_in_thread(state, **kwargs):
    import threading

    errors: list = []

    def _serve():
        try:
            daemon.run_server(state, port=0, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - surfaced to the test
            errors.append(exc)

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()
    return thread, errors


def test_idle_daemon_exits_and_unregisters(state):
    """With nothing queued or running, a daemon given an idle exit stops serving
    and removes its registration, so the next hook starts a fresh one."""
    thread, errors = _serve_in_thread(state, idle_exit_minutes=0.01)
    thread.join(timeout=30)
    assert not thread.is_alive(), "idle daemon kept serving"
    assert not errors, errors
    assert not daemon.endpoint_path(state).exists()
    assert daemon_queue.daemon_registered_alive(state) is False


def test_idle_exit_waits_for_queued_work(state, monkeypatch):
    """A queued job -- even one still inside its debounce window -- keeps the
    daemon up; it exits only after the queue drains."""
    monkeypatch.setattr("mempalace.service.execute_job", lambda kind, payload: {"success": True})
    runtime = daemon.DaemonRuntime(state)
    runtime.store.enqueue("mine", {"source": "s"}, dedupe_key="k", settle_seconds=3600)
    assert runtime.idle_seconds() == 0.0
    job = runtime.store.claim_next()
    assert job is None  # not due yet; still counted as pending work
    assert runtime.idle_seconds() == 0.0


def test_configured_idle_exit_overrides_the_autostart_default(state, monkeypatch):
    monkeypatch.setenv("MEMPALACE_DAEMON_IDLE_EXIT_MINUTES", "0")
    assert daemon._resolve_idle_exit_minutes(30) == 0.0
    monkeypatch.setenv("MEMPALACE_DAEMON_IDLE_EXIT_MINUTES", "5")
    assert daemon._resolve_idle_exit_minutes(30) == 5.0
    monkeypatch.delenv("MEMPALACE_DAEMON_IDLE_EXIT_MINUTES")
    assert daemon._resolve_idle_exit_minutes(30) == 30
    assert daemon._resolve_idle_exit_minutes(None) == 0.0


def test_registration_left_by_a_dead_daemon_with_a_reused_pid_is_not_alive(state):
    """A crashed daemon leaves its pid file; once the OS reuses that pid the
    lifetime-lock probe still shows no daemon, so hooks start a new one."""
    daemon_queue.state_dir(state).mkdir(parents=True, exist_ok=True)
    # Our own pid is certainly alive, but we do not hold daemon.lock.
    daemon_queue.pid_path(state).write_text(f"{os.getpid()}\n")
    daemon_queue.endpoint_path(state).write_text(json.dumps({"pid": os.getpid(), "port": 1}))
    assert daemon_queue.daemon_registered_alive(state) is False
    with daemon_queue.daemon_singleton(state) as won:
        assert won is True
        assert daemon_queue.daemon_registered_alive(state) is True


def test_a_refused_daemon_start_stops_hooks_from_starting_more(state, monkeypatch):
    """A daemon refused the palace writer lock (an MCP server owns it) records
    that; hooks then do not start daemons that would only refuse again, so
    prefer falls back to its direct path instead of queuing unrunnable work."""
    monkeypatch.setenv("MEMPALACE_PALACE_PATH", state)
    monkeypatch.setenv("MEMPALACE_HOOK_DAEMON_AUTOSTART", "1")
    assert hooks_cli._daemon_can_start() is True
    daemon_queue.record_start_refused(state, "writable daemon startup refused")
    assert hooks_cli._daemon_can_start() is False
    daemon_queue.clear_start_refused(state)
    assert hooks_cli._daemon_can_start() is True


def test_run_server_records_a_refused_start(state, monkeypatch):
    from mempalace import palace

    monkeypatch.setattr(daemon, "backend_requires_single_writer", lambda _b: True)

    @__import__("contextlib").contextmanager
    def held(_p):
        raise palace.MineAlreadyRunning("held by PID 1")
        yield

    monkeypatch.setattr(daemon, "mine_palace_lock", held)
    with pytest.raises(daemon.DaemonError, match="startup refused"):
        daemon.run_server(state)
    assert daemon_queue.start_recently_refused(state) is True
