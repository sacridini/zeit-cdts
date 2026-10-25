"""A helper: lends this computer to the group. It waits for a coordinator, then pulls
tiles, runs them with zeit's own command line and sends the outputs back.

While one tile is processed the next one is already being fetched. A tile comes from the
helper's cache, from shared storage (a NAS both computers see), or over the network from
the coordinator, in that order of preference.
"""

import http.client
import json
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import _group, _net, _system, _tasks
from ._tasks import format_bytes, format_seconds

HEARTBEAT = 5.0


def default_cache_dir() -> Path:
    from .._cache import cache_dir

    return cache_dir("tiles")


def default_cores() -> int:
    """The threads of ``n_jobs=-1`` (all the processors but one on a desktop, all of them in a job)."""
    from .._threads import resolve

    return resolve(-1)


def info(cores: int) -> dict:
    import platform

    return {"name": _group.machine_name(), "version": _group.version(), "code": _group.code_fingerprint(),
            "cores": cores,
            "python": platform.python_version(), "platform": sys.platform}


def parse_maps(maps: Sequence[str]) -> List[Tuple[str, str]]:
    """``SRC=DST`` pairs: a path the coordinator sees under SRC is read here under DST."""
    pairs = []
    for item in maps or ():
        if "=" not in item:
            raise ValueError(f"--map takes SRC=DST, got {item!r}")
        src, dst = item.split("=", 1)
        pairs.append((src, dst))
    return pairs


def map_path(path: str, maps: Sequence[Tuple[str, str]]) -> List[str]:
    """Where a coordinator's path may be found here: through each map, then as it is."""
    out = []
    norm = path.replace("\\", "/")
    for src, dst in maps:
        s = src.replace("\\", "/").rstrip("/")
        if norm.lower().startswith(s.lower()) and (len(norm) == len(s) or norm[len(s)] == "/"):
            rest = norm[len(s):].lstrip("/")
            out.append(os.path.join(dst, *rest.split("/")) if rest else dst)
    out.append(path)
    return out


def evict_cache(cache: Path, limit: int, keep: Path) -> None:
    """Remove the least recently used tiles until the cache holds at most ``limit`` bytes
    (never ``keep``, nor files being written)."""
    try:
        files = [p for p in cache.iterdir() if p.is_file() and p.suffix.lower() in _tasks.SUFFIXES
                 and ".part" not in p.name and ".build" not in p.name]
    except OSError:
        return
    total = sum(p.stat().st_size for p in files)
    for p in sorted(files, key=lambda p: p.stat().st_mtime):
        if total <= limit:
            break
        if p == keep:
            continue
        try:
            size = p.stat().st_size
            p.unlink()
            total -= size
        except OSError:
            pass     # in use


def _target(connect: str) -> Tuple[str, int]:
    host, _, port = connect.rpartition(":") if connect.count(":") == 1 else (connect, "", "")
    return (host or connect), int(port) if port else _group.TCP_PORT


def run_worker(*, cores: Optional[int] = None, connect: Optional[str] = None, join: Optional[str] = None,
               maps: Sequence[str] = (), cache_dir: Optional[str] = None, cache_gb: float = 20.0,
               once: bool = False, stop: Optional[threading.Event] = None,
               out: Optional[Callable[[str], None]] = None) -> None:
    """Lend this computer to the group until ``stop`` is set (or Ctrl+C). ``join``: the
    invitation code from ``zeit helpers invite``, for the first time. ``connect``: the
    coordinator's address (host or host:port), when broadcasts do not reach it. ``once``:
    return after the first coordinator is done."""
    from . import _pairing

    out = out or (lambda text: print(text, flush=True))
    stop = stop or threading.Event()
    cores = cores if cores and cores > 0 else default_cores()
    pairs = parse_maps(maps)
    if join:
        secret = _pairing.join(join, connect=connect, out=out, stop=stop)
        _group.save_secret(secret)
        out("Joined the group. From now on, just run: zeit worker")
    secret = _group.load_secret()
    if secret is None:
        raise RuntimeError("this computer is not in a group yet. On the main computer run `zeit helpers invite`, "
                           "then here: zeit worker --join <code>")
    gid = _group.group_id(secret)
    cache = Path(cache_dir) if cache_dir else default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    me = info(cores)
    backoff: Dict[str, float] = {}
    unreachable = False
    out(f"{me['name']} lends {cores} cores to the group (zeit {me['version']}). Waiting for work; Ctrl+C to stop.")
    try:
        while not stop.is_set():
            if connect:
                host, port = _target(connect)
            else:
                now = time.monotonic()
                found = _net.find({"kind": "helper", "group": gid, "name": me["name"], "cores": cores,
                                   "version": me["version"]},
                                  lambda r: r.get("kind") == "coordinator" and r.get("group") == gid,
                                  stop=stop)
                if found is None:
                    continue
                host, reply = found
                port = int(reply.get("port") or _group.TCP_PORT)
                if backoff.get(host, 0) > now:
                    stop.wait(2.0)
                    continue
            session = _net.Session(host, port)
            try:
                welcome = session.handshake(lambda salt: secret, me)
            except _net.ProtocolError as e:
                out(f"{host}: {e}")
                backoff[host] = time.monotonic() + 60
                session.close()
                if once:
                    raise
                stop.wait(30 if connect else 2)
                continue
            except OSError as e:
                session.close()
                if connect:
                    if not unreachable:
                        out(f"Cannot reach {host}:{port} ({e}); trying every 10 s")
                    unreachable = True
                    stop.wait(10)
                continue
            unreachable = False
            out(f"Working for {welcome.get('name', host)} ({host}): {welcome.get('command')}")
            job = _Job(session, cores, cache, int(cache_gb * 1e9), pairs, stop, out)
            with _system.keep_awake():
                job.run()
            if once:
                return
            out("Waiting for work.")
    except KeyboardInterrupt:
        stop.set()
        out("Stopped.")


class _BuildError(RuntimeError):
    """A tile recipe that did not build (a failure of the tile, not of the network)."""


class _Job:
    """Work for one coordinator, until it has no more tiles or goes away."""

    def __init__(self, session: _net.Session, cores: int, cache: Path, cache_bytes: int,
                 maps: Sequence[Tuple[str, str]], stop: threading.Event, out: Callable[[str], None]):
        self.session, self.cores, self.cache, self.cache_bytes = session, cores, cache, cache_bytes
        self.maps, self.stop, self.out = maps, stop, out
        self.slots = threading.Semaphore(2)          # one running, one being fetched
        self.ready: "queue.Queue" = queue.Queue()
        self.lock = threading.Lock()
        self.held: Dict[str, threading.Event] = {}   # attempt -> its cancel event
        self.ended = threading.Event()
        self.error: Optional[str] = None
        self.tmp = Path(tempfile.mkdtemp(prefix="zeit-worker-"))
        self.shared: Dict[int, str] = {}
        self.results_roots: Dict[Tuple[str, str], Optional[str]] = {}

    def run(self) -> None:
        threads = [threading.Thread(target=self._fetcher, name="zeit-worker-fetch", daemon=True),
                   threading.Thread(target=self._heartbeat, name="zeit-worker-heartbeat", daemon=True)]
        for t in threads:
            t.start()
        try:
            self._runner()
        except KeyboardInterrupt:
            self.stop.set()
            raise
        finally:
            self.ended.set()
            with self.lock:
                for event in self.held.values():
                    event.set()
            if self.stop.is_set():
                try:
                    self.session.call("POST", "/bye", {})
                except Exception:
                    pass
            for t in threads:
                t.join(timeout=10)
            self.session.close()
            shutil.rmtree(self.tmp, ignore_errors=True)
        if self.error:
            self.out(f"Lost the coordinator: {self.error}")

    def _end(self, error: Optional[str] = None) -> None:
        if error and not self.ended.is_set():
            self.error = error
            with self.lock:
                for event in self.held.values():    # the coordinator is gone: stop working for it
                    event.set()
        self.ended.set()
        self.ready.put(None)

    # -------------------------------------------------------------- fetching

    def _fetcher(self) -> None:
        s = self.session.clone()
        try:
            while not self.ended.is_set() and not self.stop.is_set():
                if not self.slots.acquire(timeout=0.5):
                    continue
                try:
                    status, reply = s.call("POST", "/next", {})
                except Exception as e:
                    self.slots.release()
                    self._end(f"{type(e).__name__}: {e}")
                    return
                if status != 200:
                    self.slots.release()
                    self._end(reply.get("error") or f"HTTP {status}")
                    return
                if reply.get("done"):
                    self.slots.release()
                    self._end()
                    return
                if "task" not in reply:
                    self.slots.release()
                    self.stop.wait(min(float(reply.get("wait") or 3), 10.0))
                    continue
                task = reply["task"]
                aid = str(task.get("attempt"))
                with self.lock:
                    self.held[aid] = threading.Event()
                try:
                    _tasks.check_args(task["command"], task["args"], received=True)
                    start = time.monotonic()
                    path, via = self._prepare(s, task)
                    self.ready.put((task, path, via, start, time.monotonic()))
                except _BuildError as e:
                    if not self.ended.is_set():
                        self.out(f"! {task['tile'].get('name')}: could not build it: {str(e).strip()[-300:]}")
                        self._report_failure(s, aid, f"could not build the tile: {e}")
                    self._release(aid)
                except Exception as e:
                    if not self.ended.is_set():
                        transfer = isinstance(e, (OSError, _net.ProtocolError, http.client.HTTPException))
                        self._report_failure(s, aid, f"could not get the tile here: {type(e).__name__}: {e}",
                                             transfer=transfer)
                    self._release(aid)
        finally:
            s.close()
            if not self.ended.is_set():
                self._end()

    def _prepare(self, s: _net.Session, task: dict) -> Tuple[Path, str]:
        tile = task["tile"]
        recipe = tile.get("recipe")
        if recipe is not None:
            from . import _sources

            _sources.check_recipe(recipe)
            key = _sources.recipe_key(recipe)     # computed here: the cache is named by what it holds
        else:
            key = str(tile["key"])
            if not key.isalnum():
                raise ValueError("bad tile key")
        suffix = Path(str(tile["name"])).suffix.lower()
        suffix = suffix if suffix in _tasks.SUFFIXES else ".tif"
        cached = self.cache / f"{key}{suffix}"     # complete: files only get there by a rename
        self._fetch_shared(s, task)
        if cached.is_file():
            os.utime(cached)
            self.out(f"> {tile['name']}: from the cache")
            return cached, "cache"
        part = self.cache / f"{key}.{task['attempt']}.part{suffix}"
        start = time.monotonic()
        try:
            if recipe is not None:
                with self.lock:
                    cancel = self.held.get(str(task["attempt"]))
                self.out(f"  {tile['name']}: building it here ({recipe['kind']})")
                ok, _, message = _sources.run_build(recipe, part, self.cores, cancel,
                                                    log_path=self.tmp / f"build-{task['attempt']}.log")
                if not ok:
                    raise _BuildError(message)
                via = "built"
            else:
                source = self._shared_source(tile)
                if source is not None:
                    shutil.copyfile(source, part)
                    via = "shared"
                else:
                    s.download(f"/tile/{task['attempt']}", part)    # checks the length it was told
                    via = "network"
            os.replace(part, cached)
        finally:
            if part.exists():
                part.unlink()
        seconds = max(time.monotonic() - start, 1e-3)
        size = cached.stat().st_size
        where = {"shared": "from shared storage", "network": "from the network", "built": "built"}[via]
        self.out(f"> {tile['name']}: {format_bytes(size)} {where} in {format_seconds(seconds)}"
                 + (f" ({format_bytes(size / seconds)}/s)" if via != "built" else ""))
        self._evict(keep=cached)
        return cached, via

    def _shared_source(self, tile: dict) -> Optional[str]:
        path = str(tile.get("path") or "")
        if not path:
            return None
        for candidate in map_path(path, self.maps):
            try:
                if (os.path.isfile(candidate) and os.path.getsize(candidate) == int(tile["size"])
                        and _tasks.probe(candidate) == tile["probe"]):
                    return candidate
            except OSError:
                continue
        return None

    def _fetch_shared(self, s: _net.Session, task: dict) -> None:
        for spec in task.get("shared", []):
            i = int(spec["index"])
            if i in self.shared:
                continue
            folder = self.tmp / "shared" / str(i)
            folder.mkdir(parents=True, exist_ok=True)
            for j, f in enumerate(spec["files"]):
                name = Path(str(f["name"])).name
                s.download(f"/shared/{i}/{j}", folder / name)
            self.shared[i] = str(folder / Path(str(spec["main"])).name)

    def _evict(self, keep: Path) -> None:
        evict_cache(self.cache, self.cache_bytes, keep)

    # -------------------------------------------------------------- running

    def _runner(self) -> None:
        s = self.session.clone()
        first = True
        waiting_since = time.monotonic()
        try:
            while True:
                try:
                    item = self.ready.get(timeout=0.5)    # a timeout keeps Ctrl+C working on Windows
                except queue.Empty:
                    continue
                if item is None:
                    return
                task, path, via, fetch_start, fetch_end = item
                aid = str(task["attempt"])
                starved = 0.0 if first else max(0.0, fetch_end - max(waiting_since, fetch_start))
                first = False
                with self.lock:
                    cancel = self.held.get(aid)
                if cancel is None or cancel.is_set() or self.ended.is_set() and self.error:
                    self._release(aid)
                    waiting_since = time.monotonic()
                    continue
                self._run_one(s, task, path, via, starved, cancel, fetch_end - fetch_start)
                self._release(aid)
                waiting_since = time.monotonic()
        finally:
            s.close()

    def _run_one(self, s: _net.Session, task: dict, path: Path, via: str, starved: float,
                 cancel: threading.Event, fetch_s: float) -> None:
        aid, name = str(task["attempt"]), task["tile"]["name"]
        folder = self.tmp / f"task-{aid}"
        folder.mkdir(parents=True, exist_ok=True)
        out_dir = folder / "out"
        shared = [(spec["option"], self.shared[int(spec["index"])]) for spec in task.get("shared", [])]
        self.out(f"  {name}: {task['command']} on {self.cores} cores")
        try:
            ok, seconds, message = _tasks.execute(task["command"], path, out_dir, task["args"], shared,
                                                  self.cores, cancel, log_path=folder / "log.txt")
            if ok:
                stats = {"compute_s": round(seconds, 3), "fetch_s": round(fetch_s, 3),
                         "starved_s": round(starved, 3), "via": via}
                placed = self._place(task, out_dir)
                if placed is not None:     # in the output folder already: only say so
                    status, reply = s.call("PUT", f"/result/{aid}", body=b"", headers={
                        "X-Zeit-Placed": placed, "X-Zeit-Stats": json.dumps(stats)})
                else:
                    archive = folder / "out.tar"
                    size = _tasks.pack(out_dir, archive)
                    with open(archive, "rb") as f:
                        status, reply = s.call("PUT", f"/result/{aid}", body=f, headers={
                            "Content-Length": str(size), "Content-Type": "application/x-tar",
                            "X-Zeit-Stats": json.dumps(stats)})
                if status == 200 and reply.get("accepted"):
                    self.out(f"< {name}: done in {format_seconds(seconds)}")
                elif status == 200:
                    self.out(f"< {name}: done in {format_seconds(seconds)}, but another computer was faster")
                else:
                    self.out(f"! {name}: the results were refused: {reply.get('error') or status}")
            elif cancel.is_set():
                if not self.ended.is_set():
                    self.out(f"  {name}: cancelled (another computer finished it)")
            else:
                last = message.strip().splitlines()[-1] if message.strip() else "failed"
                self.out(f"! {name}: failed: {last}")
                self._report_failure(s, aid, message)
        except (OSError, _net.ProtocolError) as e:
            self._end(f"{type(e).__name__}: {e}")
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    def _place(self, task: dict, out_dir: Path) -> Optional[str]:
        """Copy the outputs into the coordinator's output folder when this computer sees it
        (a NAS): the name of the folder written, or None to send them instead."""
        spec = task.get("results")
        if not isinstance(spec, dict):
            return None
        name = str(spec.get("name") or "")
        if not name or Path(name).name != name or name in (".", ".."):
            return None
        root = self._results_root(spec)
        if root is None:
            return None
        dest = Path(root) / name
        try:
            shutil.copytree(out_dir, dest)
        except OSError as e:
            shutil.rmtree(dest, ignore_errors=True)
            self.out(f"  could not write into the shared output folder ({e}); sending the results")
            return None
        return name

    def _results_root(self, spec: dict) -> Optional[str]:
        key = (str(spec.get("path")), str(spec.get("token")))
        if key not in self.results_roots:
            found = None
            probe = Path(str(spec.get("probe") or "")).name
            for candidate in map_path(key[0], self.maps) if probe else ():
                try:
                    if (Path(candidate) / probe).read_text().strip() == key[1]:
                        found = candidate
                        break
                except OSError:
                    continue
            self.results_roots[key] = found
        return self.results_roots[key]

    def _report_failure(self, s: _net.Session, aid: str, message: str, transfer: bool = False) -> None:
        try:
            s.call("POST", f"/fail/{aid}", {"error": message[-4000:], "transfer": transfer})
        except Exception:
            pass

    def _release(self, aid: str) -> None:
        with self.lock:
            self.held.pop(aid, None)
        self.slots.release()

    # -------------------------------------------------------------- heartbeat

    def _heartbeat(self) -> None:
        s = self.session.clone()
        misses = 0
        try:
            while not self.ended.wait(HEARTBEAT):
                with self.lock:
                    held = list(self.held)
                try:
                    status, reply = s.call("POST", "/heartbeat", {"held": held})
                except Exception as e:
                    misses += 1
                    if misses >= 3:
                        self._end(f"no answer ({type(e).__name__}: {e})")
                        return
                    continue
                if status == 401:
                    self._end("the coordinator no longer knows this computer")
                    return
                misses = 0
                with self.lock:
                    for aid in reply.get("cancel", []):
                        event = self.held.get(str(aid))
                        if event is not None:
                            event.set()
        finally:
            s.close()
