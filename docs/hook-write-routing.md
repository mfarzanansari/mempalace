# Hook write routing

> [!IMPORTANT]
> This page documents unreleased hook write-routing behavior on `develop`.
> MemPalace v3.6.0 does not consume `write_routing.hooks`; see the
> [v3.6.0 write-routing policy](https://github.com/MemPalace/mempalace/blob/v3.6.0/docs/write-routing-policy.md#backward-compatibility)
> for behavior available in that release.

Hook-triggered writes use the shared write-routing policy introduced for the
Tier 3 rollout tracked in #1963.

## Scope

This applies to every routine write initiated by the Python hook layer:

- Stop-hook diary checkpoints;
- transcript/conversation ingest;
- project auto-ingest;
- SessionEnd final flushes;
- PreCompact transcript ingest;
- PreCompact synchronous project mining.

It does not change CLI write routing. CLI adoption is a separate follow-up.

## Policy behavior

### `direct`

Hooks use the existing direct in-process or subprocess paths.

The daemon is not probed.

### `prefer`

Hooks use the daemon when it is healthy, and start it (detached) when none is
registered. If it cannot be used -- auto-start is off, or a registered daemon
is alive but does not answer -- hooks retain the historical direct fallback.

### `require`

Hooks use the daemon when it is healthy, and start it (detached) when none is
registered.

If the daemon cannot be used (auto-start off, or a registered daemon that does
not answer):

- no in-process ChromaDB write runs;
- no direct `mempalace mine` subprocess is started;
- the hook log records the skipped operation;
- the hook returns a visible `systemMessage`;
- the Stop save marker is not advanced, allowing a later retry.

## How hooks reach the daemon

A hook never waits on the daemon and never imports the storage stack:

1. It probes `/health` with a 150 ms timeout, using only the standard library.
2. It writes the job straight into the daemon's durable SQLite queue
   (`mempalace/daemon_queue.py`, which imports neither chromadb nor numpy).
3. If no daemon is registered, it spawns `mempalace.daemon serve` detached
   (its own session on POSIX; a hidden console and its own process group on
   Windows) and returns. The daemon's own start-up runs outside the hook budget.
   SessionStart does the same, so the daemon is usually warm by the first Stop.

Two hooks may spawn at the same moment. Every daemon takes a per-palace
lifetime lock (`daemon.lock`) before touching the palace; the loser exits with
status 0 and the winner drains the queue. Hooks also skip the spawn for
10 seconds after one, while the first daemon comes up.

Diary checkpoints are queued the same way. The daemon files the drawer and
then writes the checkpoint ack and the session's save marker, so the marker
still advances only once the entry is really stored.

A daemon a hook started exits after 30 minutes with nothing running or queued
(a debounced job not yet due counts as queued), and the next hook starts it
again. `mempalace daemon start` keeps running unless configured:

    MEMPALACE_DAEMON_IDLE_EXIT_MINUTES=30  # or daemon.idle_exit_minutes; 0 = never

### Where a daemon mine runs

The daemon is the palace's only writer. A mine job's compute (reading files,
chunking, embedding) runs in a child process at background priority: Linux
`SCHED_IDLE` (fallback `nice 19`), macOS `taskpolicy -b`, Windows background
mode (fallback idle priority class). The child never opens the storage backend;
each collection call it makes travels over a pipe and runs in the daemon, with
vectors the child already computed. A persistent Chroma database therefore
stays open in one process only.

The daemon serves those calls one at a time under a write lock that diary
checkpoints also take, and diary checkpoints have their own worker lane, so a
checkpoint lands between a long mine's batches instead of after it.

The child uses 2 ONNX Runtime intra-op threads with thread spinning off:

    MEMPALACE_DAEMON_MINE_THREADS=2        # or daemon.mine_threads; 0 = embedding_threads
    MEMPALACE_DAEMON_MINE_CHILD=0          # or daemon.mine_child: false; mine on the daemon thread

The child sends a heartbeat every 10 s, and the daemon's reaper kills a child
that has sent nothing for 60 s (time the daemon spends running one of its
calls does not count), or one still running 30 s after it reported done. The
job then fails and the daemon keeps going; writes already made stay, and the
next mine resumes from them. `/health` reports the running child under
`mine_progress` (pid, calls served, seconds since its last message).

    MEMPALACE_DAEMON_MINE_STALL_SECONDS=60  # or daemon.mine_stall_seconds; 0 = no reaper

### Turning auto-start off

Auto-start only acts under `prefer` or `require`; the default `direct` policy
never probes or starts a daemon, so daemon mode stays opt-in. For installs that
supervise the daemon themselves (a service manager, a login item):

    MEMPALACE_HOOK_DAEMON_AUTOSTART=0

or in `config.json`:

    {"hooks": {"daemon_autostart": false}}

With auto-start off, start the daemon before the first hook:

    mempalace daemon start

## One decision per hook event

A Stop or SessionEnd event may perform several writes:

1. diary checkpoint;
2. transcript ingest;
3. project auto-ingest.

The route is resolved once and stored in a context-local value for the whole
write burst. This avoids repeated health probes and prevents different writes
from selecting inconsistent routes during the same event.

## Coalescing and debounce

Daemon mine jobs are keyed by source (`hook:mine:<mode>:<path>`). The queue
keeps at most one *queued* job per key, next to at most one *running* job:

- an event for a source that is already queued replaces that job's payload
  (latest wins) and keeps its place in line;
- an event for a source that is currently running queues one pending rerun,
  so content that arrived mid-run is mined right after it, not at the next
  hook.

Each trigger has a settle window and a maximum wait. The job runs at
`MIN(first_enqueued + max_wait, last_enqueued + settle)`; an immediate trigger
is never delayed by a later debounced one.

| Trigger      | Settle | Max wait |
|--------------|--------|----------|
| Stop         | 120 s  | 15 min   |
| PreCompact   | 0      | 0        |
| SessionEnd   | 0      | 0        |

Override per trigger with `MEMPALACE_DAEMON_<TRIGGER>_SETTLE_SECONDS` /
`MEMPALACE_DAEMON_<TRIGGER>_MAX_WAIT_SECONDS` (`STOP`, `PRECOMPACT`,
`SESSION_END`), or in `config.json`:

    {
      "daemon": {
        "debounce": {
          "stop": {"settle_seconds": 120, "max_wait_seconds": 900}
        }
      }
    }

Diary checkpoints are never debounced.

## Hallways off the mine path

A mine used to finish by recomputing the hallways of the whole wing it filed
into. That pass is exact but costs tens of seconds on a wing of ~100k drawers,
and it cannot be made incremental: entity spellings are resolved across the
whole wing, so one new drawer can change how earlier drawers' entities
resolve. A hook that re-mines a transcript every few minutes paid it every
time.

Background mines now defer it:

1. The mine files its drawers, then marks the wing dirty in the hallway store
   (`hallways.sqlite3`, table `dirty_wings`) and queues one `hallways` job for
   the wing (dedupe key `hallways:<wing>`).
2. Jobs for one wing coalesce like mine jobs: the recompute runs 10 minutes
   after the wing's last mine, and never later than 60 minutes after its first
   dirty mark. A burst of mines costs one recompute.
3. The daemon runs the job in its idle-priority mine child (or its worker
   thread when `daemon.mine_child` is off), under the palace write lock, with
   the same whole-wing algorithm as before. Project mines also refresh the
   cross-wing entity tunnels derived from the wing's hallways there.

The dirty mark is what guarantees the work, not the queue row:

- A recompute clears the mark in the transaction that saves the new records,
  and only if no mine marked the wing again after it started reading drawers.
- When a `hallways` job ends with the mark still present (it failed, was
  killed, or a mine marked the wing meanwhile), the daemon queues it again.
- A daemon that starts queues every mark it finds, so marks left by a crash
  or by a deferred mine that ran with no daemon are processed. The window
  still counts from the first mark.
- `mempalace hallways --dirty` lists the marked wings;
  `mempalace hallways --rebuild --dirty` recomputes them now.

Which mines defer:

| Mine | Default |
|------|---------|
| Hook mine through the daemon (`prefer` / `require`) | deferred |
| Hook mine on the direct path (spawns `mempalace mine`) | inline |
| `mempalace mine`, direct or routed through the daemon | inline |
| MCP `mempalace_mine` | inline |

`mempalace mine --defer-hallways` / `--inline-hallways` choose per run.
`hallways.defer` in `config.json` (env `MEMPALACE_HALLWAYS_DEFER`) changes the
default: `auto` (the table above), `always` (every mine defers unless it says
`--inline-hallways`; this is how a direct-path setup defers), or `never` (every
mine recomputes inline, the behaviour before deferral). The window is
`hallways.settle_seconds` / `hallways.max_wait_seconds` (env
`MEMPALACE_HALLWAYS_SETTLE_SECONDS` / `MEMPALACE_HALLWAYS_MAX_WAIT_SECONDS`):

    {
      "hallways": {"defer": "auto", "settle_seconds": 600, "max_wait_seconds": 3600}
    }

### Staleness while a wing is dirty

Readers (`mempalace hallways`, `mempalace_list_hallways`, tunnel and graph
tools) see the wing's previous hallway records until the recompute commits.
They are stale, never missing or partial: the recompute replaces the wing's
records in one SQLite transaction. Search reads drawers, not hallways, and is
unaffected. With a daemon running, a wing's hallways lag its drawers by at
most `max_wait` (60 minutes) after the first unprocessed mine, plus the
recompute's own run time and any wait for the palace lock. With no daemon, a
deferred mark waits for the next daemon start, the next inline mine of that
wing, or `mempalace hallways --rebuild --dirty`.

## Submission ambiguity

Once a daemon submission is attempted, an error never triggers direct
fallback. The daemon may have accepted the job before the client observed the
failure; retrying directly could duplicate content.

## Invalid configuration

An explicitly invalid routing policy fails closed: hook writes are blocked and
no direct ChromaDB fallback is attempted.

An unrelated configuration read/runtime failure preserves the historical
direct-save behavior so a final checkpoint is not lost because of an
independent configuration failure.

## Backward compatibility

The default policy remains `direct`.

Legacy settings remain supported through the shared policy resolver:

- `MEMPALACE_HOOKS_DAEMON=true` maps to `prefer`;
- `hooks.daemon: true` maps to `prefer`;
- false values map to `direct`.

## Configuration examples

Prefer the daemon but permit direct fallback:

    MEMPALACE_HOOK_WRITE_ROUTING=prefer

Require the daemon and prohibit direct writers:

    MEMPALACE_HOOK_WRITE_ROUTING=require

Configuration file:

    {
      "write_routing": {
        "hooks": "require"
      }
    }
