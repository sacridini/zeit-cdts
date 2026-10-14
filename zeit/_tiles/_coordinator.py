"""The coordinator: hands tiles out to this computer and to the helpers that connect, and
gathers their outputs into the output folder.

Helpers pull: each asks for a tile when it has room (one running, one being fetched), so
fast and slow machines balance by themselves. The largest tiles go first. Near the end, an
idle machine that would finish a straggler sooner runs it too, and the first result wins.
"""

import os
import secrets
import shutil
import socket
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import _group, _net, _system, _tasks
from ._tasks import Plan, Tile, format_bytes, format_seconds

MAX_FAILURES = 2          # a tile that fails this many times is given up
MAX_TRANSFER_ERRORS = 5   # transfers of a tile that broke: back to the queue, then counted as failures
HEARTBEAT_TIMEOUT = 60.0  # a helper silent this long is gone; its tiles go back to the queue
LOST_AFTER = 20.0         # an attempt a helper does not report holding is lost
DISK_CHUNK = 16 << 20     # tiles are read one chunk at a time: sequential on a hard disk
NO_HELPERS_HINT = 20.0
EMA = 0.5
CACHE_BYTES = 20 * 10 ** 9   # tiles this computer built and keeps for the next run


@dataclass
class Machine:
    name: str
    local: bool
    cores: int
    address: str = ""
    token: str = ""
    version: str = ""
    last_seen: float = field(default_factory=time.monotonic)
    gone: bool = False
    done: int = 0
    compute_rate: Optional[float] = None    # tile bytes per second of processing
    fetch_rate: Optional[float] = None      # tile bytes per second of fetching
    compute_s: float = 0.0
    starved_s: float = 0.0
    to_cancel: set = field(default_factory=set)
    via: Dict[str, int] = field(default_factory=dict)
    builds: bool = False                    # this computer builds its tiles (recipes): it fetches too
    placed: int = 0                         # results written straight into the output folder

    @property
    def fetches(self) -> bool:
        return not self.local or self.builds

    def estimate(self, size: int) -> Optional[float]:
        if self.compute_rate is None or (self.fetches and self.fetch_rate is None):
            return None
        return (size / self.fetch_rate if self.fetches else 0.0) + size / self.compute_rate


@dataclass
class _Task:
    tile: Tile
    state: str = "pending"      # pending, running, done, failed
    attempts: List["_Attempt"] = field(default_factory=list)
    failures: int = 0
    transfer_errors: int = 0
    failed_on: set = field(default_factory=set)
    errors: List[str] = field(default_factory=list)


@dataclass
class _Attempt:
    id: str
    task: _Task
    machine: Machine
    started: float
    cancel: threading.Event = field(default_factory=threading.Event)
    closed: bool = False


@dataclass
class TileRunResult:
    """What a tile run did: the tiles finished now, those already done before, the failed
    ones (name -> error) and how many tiles each computer processed."""
    output_dir: Path
    done: List[str]
    skipped: List[str]
    failed: Dict[str, str]
    machines: Dict[str, int]
    seconds: float

    @property
    def ok(self) -> bool:
        return not self.failed


class _Printer:
    def __init__(self, out: Optional[Callable[[str], None]]):
        self._out = out
        self._tty = out is None and sys.stdout.isatty()
        self._status = ""
        self._lock = threading.Lock()

    def line(self, text: str) -> None:
        with self._lock:
            if self._out is not None:
                self._out(text)
                return
            if self._tty and self._status:
                sys.stdout.write("\r" + " " * len(self._status) + "\r")
            print(text, flush=True)
            if self._tty and self._status:
                sys.stdout.write(self._status)
                sys.stdout.flush()

    def status(self, text: str) -> None:
        with self._lock:
            if self._tty:
                pad = max(0, len(self._status) - len(text))
                sys.stdout.write("\r" + text + " " * pad)
                sys.stdout.flush()
                self._status = text
            elif self._out is None:
                print(text, flush=True)
            else:
                self._out(text)

    def end(self) -> None:
        with self._lock:
            if self._tty and self._status:
                sys.stdout.write("\n")
                sys.stdout.flush()
            self._status = ""

    @property
    def interval(self) -> float:
        return 1.0 if self._tty else 30.0


class _Limited:
    """A request body as a file of exactly Content-Length bytes."""

    def __init__(self, stream, length: int):
        self._stream, self._left = stream, length

    def read(self, n: int = -1) -> bytes:
        if self._left <= 0:
            return b""
        n = self._left if n is None or n < 0 else min(n, self._left)
        data = self._stream.read(n)
        self._left -= len(data)
        return data


class Coordinator:
    def __init__(self, plan: Plan, *, local: bool = True, local_cores: Optional[int] = None,
                 helpers: bool = False, port: int = _group.TCP_PORT, discovery: bool = True,
                 out: Optional[Callable[[str], None]] = None, host: str = ""):
        if not local and not helpers:
            raise ValueError("nothing would process the tiles: use this computer (local) or helpers, or both")
        self.plan = plan
        self.helpers, self.discovery, self.port, self.host = helpers, discovery, port, host
        self.version = _group.version()
        self.name = _group.machine_name()
        self._print = _Printer(out)
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self.finished = threading.Event()
        self._stop = threading.Event()
        self._disk = threading.Lock()
        self._tasks = [_Task(t) for t in plan.tiles]
        self._attempts: Dict[str, _Attempt] = {}
        self._machines: List[Machine] = []
        self._by_token: Dict[str, Machine] = {}
        self._counter = 0
        self._sent = deque()             # (time, bytes) of tile data sent, for the rate
        self._warned = set()
        self._started = time.monotonic()
        self._threads: List[threading.Thread] = []
        self._server = self._responder = None
        self._tmp: Optional[tempfile.TemporaryDirectory] = None
        self.local_machine: Optional[Machine] = None
        if local:
            cores = local_cores if local_cores and local_cores > 0 else max(1, (os.cpu_count() or 2) - 1)
            self.local_machine = Machine(f"{self.name} (this computer)", True, cores, builds=plan.recipes)
            self._machines.append(self.local_machine)
        self.compress_transfers = True
        self._sending: Dict[int, Path] = {}     # tile id -> its compressed copy for the network
        self._send_locks: Dict[int, threading.Lock] = {}
        self._results: Optional[dict] = None    # where helpers that see the output folder put results
        self._shared = [_tasks.shared_files(s["path"]) for s in plan.shared]
        if not self._tasks:
            self.finished.set()

    # -------------------------------------------------------------- lifecycle

    def start(self) -> "Coordinator":
        if self.helpers and not self.finished.is_set():
            secret = _group.load_secret()
            if secret is None:
                raise RuntimeError("this computer is not in a group of helpers yet: run `zeit helpers invite` "
                                   "here, then `zeit worker --join <code>` on each helper")
            self._gid = _group.group_id(secret)
            self._tmp = tempfile.TemporaryDirectory(prefix="zeit-tiles-")
            cert, key, fingerprint = _group.certificate(Path(self._tmp.name))
            self._handshakes = _net.Handshakes(secret, fingerprint)
            self._server = _net.Server(self, self.port, cert, key, host=self.host).start()
            self.port = self._server.port
            if self.discovery:
                try:
                    self._responder = _net.Responder(self._answer).start()
                except OSError as e:
                    self._print.line(f"Discovery is off (UDP port {_group.UDP_PORT}: {e}); helpers can still "
                                     f"connect with: zeit worker --connect {self._address()}:{self.port}")
            self._results = self._results_probe()
            if not self.plan.recipes:
                self._warn_formats()
        if self.local_machine is not None and not self.finished.is_set():
            self._spawn(self._local_loop, "zeit-tiles-local")
        self._spawn(self._monitor, "zeit-tiles-monitor")
        return self

    def _spawn(self, target, name) -> None:
        thread = threading.Thread(target=target, name=name, daemon=True)
        thread.start()
        self._threads.append(thread)

    def wait(self) -> TileRunResult:
        try:
            with _system.keep_awake():
                while not self.finished.wait(0.5):
                    pass
                time.sleep(0.5 if self.helpers else 0)   # helpers asking now hear "done"
        except KeyboardInterrupt:
            self._print.end()
            self._print.line("Stopped. Run the same command again to continue: finished tiles are kept.")
            self.close()
            raise
        self.close()
        return self._result()

    def close(self) -> None:
        self._stop.set()
        self.finished.set()
        self._wake.set()
        with self._lock:
            for attempt in self._attempts.values():
                attempt.cancel.set()
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(timeout=10)
        if self._responder is not None:
            self._responder.close()
            self._responder = None
        if self._server is not None:
            self._server.close()
            self._server = None
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None
        _tasks.clean_partial(self.plan)
        self._print.end()

    def _address(self) -> str:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("192.0.2.1", 9))     # no packet is sent; picks the outgoing interface
                return s.getsockname()[0]
        except OSError:
            return socket.gethostbyname(socket.gethostname())

    def _same_network(self, address: str) -> bool:
        if address.startswith("127."):
            return True
        mine = self._address()
        return mine.rsplit(".", 1)[0] == address.rsplit(".", 1)[0]

    def _answer(self, message: dict, addr) -> Optional[dict]:
        if message.get("kind") == "helper" and message.get("group") == self._gid and not self.finished.is_set():
            return {"kind": "coordinator", "group": self._gid, "port": self.port, "name": self.name,
                    "command": self.plan.command}
        return None

    # -------------------------------------------------------------- scheduling

    def _new_attempt(self, task: _Task, machine: Machine) -> _Attempt:
        self._counter += 1
        attempt = _Attempt(f"{task.tile.id}.{self._counter}", task, machine, time.monotonic())
        task.attempts.append(attempt)
        task.state = "running"
        self._attempts[attempt.id] = attempt
        return attempt

    def _next(self, machine: Machine) -> Tuple[str, Any]:
        """("task", attempt), ("wait", seconds) or ("done", None)."""
        with self._lock:
            if self.finished.is_set() or self._stop.is_set():
                return "done", None
            pending = [t for t in self._tasks if t.state == "pending"]
            if pending:
                task = ([t for t in pending if machine.name not in t.failed_on] or pending)[0]
                if not machine.local and self._local_sooner(task, machine, len(pending)):
                    return "wait", 2.0
                return "task", self._new_attempt(task, machine)
            task = self._straggler(machine)
            if task is not None:
                return "task", self._new_attempt(task, machine)
            return "wait", 3.0

    def _remaining(self, machine: Machine, now: float) -> float:
        """Estimated seconds until ``machine`` is free (0 when idle or unknown)."""
        left = 0.0
        for a in self._attempts.values():
            if a.machine is machine and not a.closed:
                est = machine.estimate(a.task.tile.size)
                if est is not None:
                    left = max(left, a.started + est - now)
        return left

    def _local_sooner(self, task: _Task, machine: Machine, n_pending: int) -> bool:
        """The last tile stays here when this computer would finish it before the helper."""
        local = self.local_machine
        if local is None or n_pending > 1 or local.compute_rate is None:
            return False
        estimate = machine.estimate(task.tile.size)
        if estimate is None:
            return False
        now = time.monotonic()
        theirs = self._remaining(machine, now) + estimate    # a helper asking ahead is still busy
        ours = self._remaining(local, now) + local.estimate(task.tile.size)
        return ours < theirs

    def _straggler(self, machine: Machine) -> Optional[_Task]:
        """A running tile that ``machine`` would finish clearly sooner than its holder."""
        now = time.monotonic()
        if self._remaining(machine, now) > 0:
            return None
        best, best_finish = None, None
        for task in self._tasks:
            if task.state != "running" or any(a.machine is machine for a in task.attempts):
                continue
            active = [a for a in task.attempts if not a.closed]
            if len(active) != 1:
                continue
            holder = active[0]
            est_holder = holder.machine.estimate(task.tile.size)
            est_mine = machine.estimate(task.tile.size)
            if est_holder is None or est_mine is None:
                continue
            finish = holder.started + est_holder
            margin = max(30.0, 0.1 * (finish - now))
            if now + est_mine < finish - margin and (best_finish is None or finish > best_finish):
                best, best_finish = task, finish
        return best

    def _close(self, attempt: _Attempt) -> None:
        attempt.closed = True
        attempt.cancel.set()
        self._attempts.pop(attempt.id, None)

    def _complete(self, attempt: _Attempt, folder: Path, stats: dict) -> bool:
        with self._lock:
            task, machine = attempt.task, attempt.machine
            if attempt.closed or task.state in ("done", "failed"):
                self._close(attempt)
                return False
            try:
                _tasks.finalize(self.plan, task.tile, folder, {
                    "machine": machine.name, "seconds": round(float(stats.get("compute_s") or 0), 1)})
            except OSError as e:
                self._fail(attempt, f"could not move the outputs into place: {e}")
                return False
            self._close(attempt)
            task.state = "done"
            self._drop_sending(task)
            for other in task.attempts:
                if not other.closed:
                    self._close(other)
                    if not other.machine.local:
                        other.machine.to_cancel.add(other.id)
            size = task.tile.size
            compute_s = max(float(stats.get("compute_s") or 0), 1e-3)
            machine.compute_rate = _ema(machine.compute_rate, size / compute_s)
            fetch_s = float(stats.get("fetch_s") or 0)
            if machine.fetches and fetch_s > 0.05:
                machine.fetch_rate = _ema(machine.fetch_rate, size / fetch_s)
            elif machine.fetches and machine.fetch_rate is None:
                machine.fetch_rate = float("inf")
            machine.done += 1
            machine.compute_s += compute_s
            machine.starved_s += float(stats.get("starved_s") or 0)
            via = str(stats.get("via") or "local")
            machine.via[via] = machine.via.get(via, 0) + 1
            if stats.get("results") == "shared":
                machine.placed += 1
            self._check_finished()
        self._wake.set()
        return True

    def _fail(self, attempt: _Attempt, message: str) -> None:
        with self._lock:
            task, machine = attempt.task, attempt.machine
            if attempt.closed:
                return
            self._close(attempt)
            task.failures += 1
            task.failed_on.add(machine.name)
            task.errors.append(f"{machine.name}: {message}")
            if task.state in ("done", "failed") or any(not a.closed for a in task.attempts):
                return
            task.state = "failed" if task.failures >= MAX_FAILURES else "pending"
            if task.state == "failed":
                self._drop_sending(task)
            first = message.strip().splitlines()[-1] if message.strip() else "no message"
            again = "given up" if task.state == "failed" else "will try again"
            self._print.line(f"! {task.tile.stem} failed on {machine.name} ({again}): {first}")
            self._check_finished()
        self._wake.set()

    def _lose(self, attempt: _Attempt, transfer_error: Optional[str] = None) -> None:
        """An attempt that will never report (its helper left, or a transfer broke): the tile
        goes back to the queue, without counting as a failure of the tile, unless its
        transfers keep breaking."""
        with self._lock:
            if attempt.closed:
                return
            if transfer_error is not None:
                attempt.task.transfer_errors += 1
                if attempt.task.transfer_errors > MAX_TRANSFER_ERRORS:
                    self._fail(attempt, transfer_error)
                    return
            self._close(attempt)
            task = attempt.task
            if task.state == "running" and not any(not a.closed for a in task.attempts):
                task.state = "pending"
        self._wake.set()

    def _check_finished(self) -> None:
        if all(t.state in ("done", "failed") for t in self._tasks):
            self.finished.set()
            self._wake.set()

    # -------------------------------------------------------------- this computer

    def _local_loop(self) -> None:
        machine = self.local_machine
        shared = [(s["option"], s["path"]) for s in self.plan.shared]
        while not self.finished.is_set():
            kind, attempt = self._next(machine)
            if kind != "task":
                self._wake.wait(1.0)
                self._wake.clear()
                continue
            tile = attempt.task.tile
            folder = _tasks.partial_dir(self.plan, f"{tile.stem}-{attempt.id}")
            out = folder / "out"
            stats = {"via": "local"}
            try:
                path = tile.path
                if tile.recipe is not None:
                    path, stats["fetch_s"], stats["via"] = self._build_here(tile, attempt, folder)
                ok, seconds, message = _tasks.execute(self.plan.command, path, out, self.plan.args, shared,
                                                      machine.cores, attempt.cancel, log_path=folder / "log.txt")
            except Exception as e:
                ok, seconds, message = False, 0.0, f"{type(e).__name__}: {e}"
            if ok:
                self._complete(attempt, out, {**stats, "compute_s": seconds})
            elif attempt.cancel.is_set():
                self._lose(attempt)
            else:
                self._fail(attempt, message)
            shutil.rmtree(folder, ignore_errors=True)

    def _build_here(self, tile: Tile, attempt: _Attempt, folder: Path) -> Tuple[Path, float, str]:
        """A recipe tile on this computer: from the cache, or built into it."""
        from . import _sources
        from ._worker import default_cache_dir, evict_cache

        cache = default_cache_dir()
        cache.mkdir(parents=True, exist_ok=True)
        cached = cache / f"{tile.key}.tif"
        if cached.is_file():
            os.utime(cached)
            return cached, 0.0, "cache"
        part = cache / f"{tile.key}.{attempt.id}.build.tif"
        try:
            ok, seconds, message = _sources.run_build(tile.recipe, part, self.local_machine.cores, attempt.cancel,
                                                      log_path=folder / "build.log")
            if not ok:
                raise RuntimeError(f"could not build the tile: {message}")
            os.replace(part, cached)
        finally:
            if part.exists():
                part.unlink()
        evict_cache(cache, CACHE_BYTES, keep=cached)
        return cached, seconds, "built"

    # -------------------------------------------------------------- the helpers' requests

    def handle(self, handler, method: str, path: str) -> None:
        route = path.split("?", 1)[0].rstrip("/")
        if method == "POST" and route == "/hello":
            handler.send_json(200, self._handshakes.hello(handler.read_json()))
            return
        if method == "POST" and route == "/auth":
            self._auth(handler)
            return
        machine = self._by_token.get(handler.headers.get("X-Zeit-Session") or "")
        if machine is None or machine.gone:
            handler.send_json(401, {"error": "not authenticated"})
            return
        machine.last_seen = time.monotonic()
        parts = route.strip("/").split("/")
        if method == "POST" and route == "/next":
            handler.read_body()
            self._serve_next(handler, machine)
        elif method == "GET" and parts[0] == "tile" and len(parts) == 2:
            self._serve_tile(handler, machine, parts[1])
        elif method == "GET" and parts[0] == "shared" and len(parts) == 3:
            self._serve_shared(handler, int(parts[1]), int(parts[2]))
        elif method == "POST" and route == "/heartbeat":
            self._heartbeat(handler, machine)
        elif method == "PUT" and parts[0] == "result" and len(parts) == 2:
            self._receive(handler, machine, parts[1])
        elif method == "POST" and parts[0] == "fail" and len(parts) == 2:
            body = handler.read_json()
            attempt = self._attempts.get(parts[1])
            if attempt is not None and attempt.machine is machine:
                message = str(body.get("error") or "failed")[:4000]
                if body.get("transfer"):
                    self._lose(attempt, transfer_error=message)
                else:
                    self._fail(attempt, message)
            handler.send_json(200, {})
        elif method == "POST" and route == "/bye":
            handler.read_body()
            self._leave(machine, "left")
            handler.send_json(200, {})
        else:
            handler.send_json(404, {"error": f"no {method} {route}"})

    def _auth(self, handler) -> None:
        body = handler.read_json()
        if not self._handshakes.auth(body):
            handler.send_json(403, {"error": "not in this group"})
            return
        info = body.get("info") if isinstance(body.get("info"), dict) else {}
        version = str(info.get("version", "?"))
        name = "".join(c for c in str(info.get("name") or "helper") if c.isprintable())[:64] or "helper"
        if version != self.version:
            self._warn_once(f"version:{name}",
                            f"! {name} has zeit {version}; this computer has {self.version}. It will not get tiles "
                            f"until both have the same version (pip install -U zeit-cdts=={self.version}).")
            handler.send_json(409, {"error": f"this computer has zeit {version}; {self.name} uses {self.version}. "
                                             f"Update with: pip install zeit-cdts=={self.version}",
                                    "version": self.version})
            return
        if str(info.get("code")) != _group.code_fingerprint():
            self._warn_once(f"code:{name}",
                            f"! {name} has zeit {version} too, but with other code (a development install on one "
                            "of them?). It will not get tiles until both have the same code.")
            handler.send_json(409, {"error": f"this computer has zeit {version} with other code than {self.name} "
                                             "(a development install on one of them?): install the same zeit on both"})
            return
        address = handler.client_address[0]
        with self._lock:
            for old in self._machines:
                if not old.local and not old.gone and old.name == name and old.address == address:
                    self._leave(old, "reconnected")     # a helper that restarted
            taken = {m.name for m in self._machines if not m.gone}
            unique, n = name, 2
            while unique in taken:
                unique, n = f"{name}#{n}", n + 1
            try:
                cores = max(1, int(info.get("cores") or 1))
            except (TypeError, ValueError):
                cores = 1
            machine = Machine(unique, False, cores, address=address, token=secrets.token_hex(32), version=version)
            self._machines.append(machine)
            self._by_token[machine.token] = machine
        self._print.line(f"+ {machine.name} ({address}, {cores} cores) joined")
        if self.plan.recipes and self._same_network(address):
            self._warn_once("shared-link",
                            f"! {machine.name} is on this computer's local network, so it most likely shares its "
                            "internet link: building tiles there is faster only if one computer cannot fill the "
                            "link alone. (Earth Engine also limits concurrent requests per project, not per "
                            "computer.)")
        handler.send_json(200, {"token": machine.token, "name": self.name, "command": self.plan.command})

    def _serve_next(self, handler, machine: Machine) -> None:
        kind, value = self._next(machine)
        if kind == "done":
            handler.send_json(200, {"done": True})
        elif kind == "wait":
            handler.send_json(200, {"wait": value})
        else:
            handler.send_json(200, {"task": self._payload(value)})

    def _payload(self, attempt: _Attempt) -> dict:
        tile = attempt.task.tile
        shared = []
        for i, (spec, files) in enumerate(zip(self.plan.shared, self._shared)):
            shared.append({"option": spec["option"], "index": i, "main": Path(spec["path"]).name,
                           "files": [{"name": Path(f).name, "size": os.path.getsize(f)} for f in files]})
        if tile.recipe is not None:
            about = {"name": tile.path.name, "key": tile.key, "recipe": tile.recipe}
        else:
            if not tile.probe:
                tile.probe = _tasks.probe(tile.path, tile.size)
                tile.shared_path = _tasks.network_path(tile.path)
            about = {"name": tile.path.name, "size": tile.size, "key": tile.key, "probe": tile.probe,
                     "path": tile.shared_path}
        payload = {"attempt": attempt.id, "command": self.plan.command, "args": self.plan.args, "shared": shared,
                   "tile": about}
        if self._results is not None:
            payload["results"] = {**self._results, "name": self._placed_name(attempt)}
        return payload

    def _placed_name(self, attempt: _Attempt) -> str:
        return f"{attempt.task.tile.stem}-{attempt.id}-placed"

    def _results_probe(self) -> Optional[dict]:
        """A file in the output folder by which a helper that sees the folder too (a NAS)
        recognizes it, and then writes its results there instead of sending them."""
        root = self.plan.output_dir / _tasks.PARTIAL
        try:
            root.mkdir(parents=True, exist_ok=True)
            token = secrets.token_hex(16)
            name = f"probe-{token[:8]}.txt"
            (root / name).write_text(token)
        except OSError:
            return None
        return {"path": _tasks.network_path(root), "probe": name, "token": token}

    def _serve_tile(self, handler, machine: Machine, attempt_id: str) -> None:
        attempt = self._attempts.get(attempt_id)
        if attempt is None or attempt.machine is not machine or attempt.closed:
            handler.send_json(404, {"error": "no such task"})
            return
        tile = attempt.task.tile
        if tile.recipe is not None:
            handler.send_json(404, {"error": "this tile is built by the helper"})
            return
        try:
            path = self._to_send(tile)
        except Exception as e:
            self._warn_once(f"compress:{tile.id}", f"! Could not compress {tile.path.name} for the network ({e}); "
                                                   "sending it as it is.")
            path = tile.path
        size = os.path.getsize(path)

        def chunks():
            with open(path, "rb") as f:
                while True:
                    with self._disk:       # one read at a time: sequential on a hard disk
                        data = f.read(DISK_CHUNK)
                    if not data:
                        return
                    for i in range(0, len(data), _net.CHUNK):
                        piece = data[i:i + _net.CHUNK]
                        yield piece
                        self._count_sent(len(piece))

        handler.send_stream(size, chunks())

    def _to_send(self, tile: Tile) -> Path:
        """The file to send: the tile itself, or a losslessly compressed copy of a tile
        stored without compression (the same values, a fraction of the bytes on the wire)."""
        if not self.compress_transfers or tile.compressed:
            return tile.path
        with self._lock:
            lock = self._send_locks.setdefault(tile.id, threading.Lock())
        with lock:
            if tile.id in self._sending and self._sending[tile.id].exists():
                return self._sending[tile.id]
            if tile.compressed is None:
                tile.compressed = _tasks.is_compressed(tile.path)
            if tile.compressed:
                return tile.path
            folder = self.plan.output_dir / _tasks.PARTIAL / "send"
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / f"{tile.stem}.tif"
            start = time.monotonic()
            _tasks.compress_copy(tile.path, target)
            size = os.path.getsize(target)
            self._print.line(f"  {tile.path.name}: compressed for the network, {format_bytes(tile.size)} -> "
                             f"{format_bytes(size)} in {format_seconds(time.monotonic() - start)}")
            self._sending[tile.id] = target
            return target

    def _drop_sending(self, task: _Task) -> None:
        path = self._sending.pop(task.tile.id, None)
        if path is not None:
            try:
                path.unlink()
            except OSError:
                pass

    def _serve_shared(self, handler, i: int, j: int) -> None:
        try:
            path = self._shared[i][j]
        except IndexError:
            handler.send_json(404, {"error": "no such file"})
            return
        data = Path(path).read_bytes()
        handler.send_stream(len(data), [data])

    def _heartbeat(self, handler, machine: Machine) -> None:
        body = handler.read_json()
        held = {str(a) for a in body.get("held", [])}
        now = time.monotonic()
        lost = []
        with self._lock:
            for attempt in list(self._attempts.values()):
                if attempt.machine is machine and attempt.id not in held and now - attempt.started > LOST_AFTER:
                    lost.append(attempt)
            cancel = sorted(machine.to_cancel & held)
            machine.to_cancel &= held
        for attempt in lost:
            self._lose(attempt)
        handler.send_json(200, {"cancel": cancel})

    def _receive(self, handler, machine: Machine, attempt_id: str) -> None:
        import json

        length = int(handler.headers.get("Content-Length") or 0)
        body = _Limited(handler.rfile, length)
        attempt = self._attempts.get(attempt_id)
        if attempt is None or attempt.machine is not machine or attempt.closed:
            while body.read(_net.CHUNK):      # drain: keep the connection usable
                pass
            handler.send_json(200, {"accepted": False})
            return
        try:
            stats = json.loads(handler.headers.get("X-Zeit-Stats") or "{}")
        except ValueError:
            stats = {}
        stats = stats if isinstance(stats, dict) else {}
        placed = handler.headers.get("X-Zeit-Placed")
        if placed is not None:
            while body.read(_net.CHUNK):
                pass
            folder = self.plan.output_dir / _tasks.PARTIAL / self._placed_name(attempt)
            if placed != self._placed_name(attempt) or not _tasks.check_placed(folder):
                shutil.rmtree(folder, ignore_errors=True)
                self._lose(attempt, transfer_error="the results placed in the output folder are not there")
                handler.send_json(200, {"accepted": False})
                return
            accepted = self._complete(attempt, folder, {**stats, "results": "shared"})
            shutil.rmtree(folder, ignore_errors=True)
            handler.send_json(200, {"accepted": accepted})
            return
        folder = _tasks.partial_dir(self.plan, f"{attempt.task.tile.stem}-{attempt.id}")
        out = folder / "out"
        try:
            _tasks.unpack(body, out)
        except Exception as e:
            shutil.rmtree(folder, ignore_errors=True)
            if isinstance(e, ValueError):      # refused contents
                while body.read(_net.CHUNK):
                    pass
                self._fail(attempt, f"bad results: {e}")
                handler.send_json(200, {"accepted": False})
            else:                              # the upload broke
                self._lose(attempt, transfer_error=f"the results did not arrive: {type(e).__name__}: {e}")
                handler.close_connection = True
            return
        accepted = self._complete(attempt, out, stats if isinstance(stats, dict) else {})
        shutil.rmtree(folder, ignore_errors=True)
        handler.send_json(200, {"accepted": accepted})

    def _leave(self, machine: Machine, why: str) -> None:
        with self._lock:
            if machine.gone:
                return
            machine.gone = True
            self._by_token.pop(machine.token, None)
            attempts = [a for a in self._attempts.values() if a.machine is machine]
        for attempt in attempts:
            self._lose(attempt)
        if why != "reconnected" and not self.finished.is_set():
            back = f"; its {len(attempts)} tile(s) go back to the queue" if attempts else ""
            self._print.line(f"- {machine.name} {why}{back}")

    # -------------------------------------------------------------- watching

    def _count_sent(self, n: int) -> None:
        with self._lock:
            self._sent.append((time.monotonic(), n))

    def _send_rate(self) -> float:
        now = time.monotonic()
        with self._lock:
            while self._sent and now - self._sent[0][0] > 10.0:
                self._sent.popleft()
            total = sum(n for _, n in self._sent)
        return total / 10.0

    def _monitor(self) -> None:
        last_status = 0.0
        while not self.finished.wait(1.0):
            now = time.monotonic()
            for machine in list(self._machines):
                if not machine.local and not machine.gone and now - machine.last_seen > HEARTBEAT_TIMEOUT:
                    self._leave(machine, "stopped responding")
            self._check_warnings(now)
            if now - last_status >= self._print.interval:
                self._print.status(self._status_line(now))
                last_status = now

    def _status_line(self, now: float) -> str:
        with self._lock:
            done = sum(t.state == "done" for t in self._tasks)
            failed = sum(t.state == "failed" for t in self._tasks)
            total = len(self._tasks)
            active = [m for m in self._machines if not m.gone]
            remaining = sum(t.tile.size for t in self._tasks if t.state in ("pending", "running"))
            throughput = 0.0
            for m in active:
                per_byte = (0.0 if m.local or not m.fetch_rate else 1.0 / m.fetch_rate) + \
                           (1.0 / m.compute_rate if m.compute_rate else 0.0)
                if m.compute_rate:
                    throughput += 1.0 / per_byte
        text = f"{self.plan.command}: {done}/{total} tiles"
        if failed:
            text += f" ({failed} failed)"
        text += f" | {len(active)} computer{'s' if len(active) != 1 else ''}"
        rate = self._send_rate()
        if rate > 0:
            text += f" | sending {format_bytes(rate)}/s"
        if throughput > 0 and remaining:
            text += f" | ~{format_seconds(remaining / throughput)} left"
        return text

    def _warn_once(self, key: str, text: str) -> None:
        with self._lock:
            if key in self._warned:
                return
            self._warned.add(key)
        self._print.line(text)

    def _check_warnings(self, now: float) -> None:
        if not self.helpers:
            return
        remote = [m for m in self._machines if not m.local]
        if not remote and now - self._started > NO_HELPERS_HINT:
            only = (" Nothing is processed until one joins (--no-local)." if self.local_machine is None else "")
            self._warn_once("no-helpers",
                            f"No helper has joined yet.{only} On each helper run: zeit worker\n"
                            f"  If they do not appear: run `zeit helpers firewall` here (Windows), or start them "
                            f"with: zeit worker --connect {self._address()}:{self.port}")
        done = [m for m in remote if m.via.get("network")]
        n = sum(m.via.get("network", 0) for m in done)
        busy = sum(m.compute_s + m.starved_s for m in done)
        starved = sum(m.starved_s for m in done)
        if n >= 3 and busy > 0 and starved / busy > 0.25:
            self._warn_once("network",
                            f"! The network is the bottleneck: the helpers waited for tiles {100 * starved / busy:.0f}% "
                            f"of the time (sending at ~{format_bytes(self._send_rate())}/s). More helpers will not "
                            "make this faster; smaller tiles will (e.g. int16 with ZSTD compression), and so will "
                            "wired instead of Wi-Fi connections.")

    def _warn_formats(self) -> None:
        if not self._tasks:
            return
        try:
            import rasterio

            with rasterio.open(self._tasks[0].tile.path) as src:
                dtype, compression = src.dtypes[0], src.compression
        except Exception:
            return
        if compression is None and dtype in ("float32", "float64"):
            size = self._tasks[0].tile.size
            self._print.line(
                f"! The tiles are {dtype} without compression ({format_bytes(size)} each). They are compressed "
                "without loss for the network (about 1.4x smaller for Landsat NBR); stored as int16 with ZSTD "
                "they would be about 3x smaller (an index scaled by 10000, at the cost of 4 decimal places).")

    def _result(self) -> TileRunResult:
        with self._lock:
            done = [t.tile.stem for t in self._tasks if t.state == "done"]
            failed = {t.tile.stem: (t.errors[-1] if t.errors else "failed") for t in self._tasks
                      if t.state == "failed"}
            machines = {m.name: m.done for m in self._machines if m.done}
        return TileRunResult(self.plan.output_dir, done, [t.stem for t in self.plan.done], failed, machines,
                             time.monotonic() - self._started)


def _ema(old: Optional[float], new: float) -> float:
    return new if old is None or old == float("inf") else EMA * new + (1 - EMA) * old


def summary(result: TileRunResult) -> str:
    lines = [f"{len(result.done)} tile(s) processed in {format_seconds(result.seconds)}"
             + (f", {len(result.skipped)} already done before" if result.skipped else "")
             + (f", {len(result.failed)} failed" if result.failed else "") + f" -> {result.output_dir}"]
    if len(result.machines) > 1:
        lines.append("  " + ", ".join(f"{name}: {n}" for name, n in result.machines.items()))
    for stem, error in result.failed.items():
        lines.append(f"  failed: {stem}: {error.strip().splitlines()[-1] if error.strip() else error}")
    return "\n".join(lines)
