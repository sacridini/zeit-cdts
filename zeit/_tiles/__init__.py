"""Many tiles, on this computer and on helpers in the local network, or split among the
tasks of a cluster's job array.

``zeit.run_tiles`` (or ``zeit <command> <folder> <output> --tiles``) runs a command on every
tile of a folder, here and on the computers running ``zeit worker`` in the same group.
Finished tiles are kept, so running it again continues where it stopped.
"""

import os
import signal
import threading
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Union

from . import _group
from ._coordinator import Coordinator, Stopped, TileRunResult, summary
from ._sources import TileSources, gee_tiles, stac_tiles, tile_urls, url_list
from ._tasks import format_bytes, make_plan, option_values

__all__ = ["run_tiles", "run_worker", "invite_helpers", "tile_urls", "stac_tiles", "gee_tiles", "TileSources",
           "TileRunResult", "Stopped", "fetch_tiles"]

JOB_VARIABLES = ("SLURM_JOB_ID", "PBS_JOBID", "LSB_JOBID")


def run_tiles(command: str, inputs: Union[str, Path, Iterable], output_dir: Union[str, Path], *,
              helpers: bool = False, local: bool = True, local_cores: Optional[int] = None,
              port: int = _group.TCP_PORT, compress_transfers: bool = True, verbose: bool = True,
              shard: Optional[str] = None, **options) -> TileRunResult:
    """Run ``command`` on every tile, writing ``<output_dir>/<tile name>/``.

    Parameters
    ----------
    command
        A command of the zeit command line that reads one raster: ``"landtrendr"``,
        ``"ccdc"``, ``"bfast_monitor"``, ``"mann_kendall"``, ``"phenology"``, ...
    inputs
        A folder (its ``.tif`` files), a glob pattern (``"tiles/LT_*.tif"``), a list of files,
        a text file of URLs (one per line), or tiles that each computer fetches or builds for
        itself: ``zeit.tile_urls``, ``zeit.stac_tiles``, ``zeit.gee_tiles``.
    output_dir
        One folder per tile is written in it. Tiles already finished there with the same
        options are skipped, so an interrupted run continues where it stopped.
    helpers
        Also send tiles to the computers of this computer's group running ``zeit worker``
        (see ``zeit helpers invite``). They read a tile from a shared folder (a NAS) when
        they see the same file, otherwise this computer sends it.
    local
        Process tiles on this computer too. ``local=False`` (with ``helpers=True``): this
        computer only hands out tiles and collects the results.
    local_cores
        Cores for the tiles processed here (default: all but one).
    compress_transfers
        Send a tile stored without compression as a losslessly compressed copy (ZSTD with a
        predictor): the same values, a fraction of the bytes on the network.
    shard
        ``"i/n"``: process part ``i`` (from 0) of ``n`` of the tiles, tile ``k`` (in the
        order of the inputs) being part ``k mod n``'s. Inside a Slurm job array the part is
        the task's without it: run the same call in every task (see ``zeit slurm``).
    **options
        The command's options, as keyword arguments (``max_segments=6``, ``event_type="gain"``).

    Returns
    -------
    TileRunResult
        The tiles done now, those skipped as already done, the failed ones (with the error)
        and how many each computer processed.

    Raises
    ------
    Stopped
        On SIGTERM (a scheduler's warning before a job's time limit, ``#SBATCH
        --signal=B:TERM@120``): the tiles being processed are dropped, the finished ones kept.

    Examples
    --------
    >>> zeit.run_tiles("landtrendr", "nbr_tiles/", "lt_out/", helpers=True, max_segments=6)
    >>> zeit.run_tiles("landtrendr", "nbr_tiles/", "lt_out/", helpers=True, local=False)   # only coordinate
    """
    from ._shard import resolve as resolve_shard

    out: Optional[Callable[[str], None]] = None if verbose else (lambda text: None)
    command = command.replace("_", "-")
    if helpers and any(os.environ.get(v) for v in JOB_VARIABLES):
        raise RuntimeError("helpers are for the computers of a local network, which find each other by broadcasts "
                           "that a cluster's network does not carry; in a cluster, split the tiles among the tasks "
                           "of a job array instead (`zeit slurm` writes the script)")
    inputs = url_list(inputs)
    part = resolve_shard(output_dir, shard)
    plan = make_plan(command, inputs, output_dir, option_values(command, options), shard=part)
    say = out or (lambda text: print(text, flush=True))
    which = f" (part {part.index} of {part.count}, {part.source})" if part is not None else ""
    if not plan.tiles:
        say(f"All {len(plan.done)} tiles{which} are already done in {plan.output_dir}.")
        return TileRunResult(plan.output_dir, [], [t.stem for t in plan.done], {}, {}, 0.0)
    coordinator = Coordinator(plan, local=local, local_cores=local_cores, helpers=helpers, port=port, out=out)
    coordinator.compress_transfers = compress_transfers
    total = sum(t.size for t in plan.tiles)
    done = f" ({len(plan.done)} already done)" if plan.done else ""
    size = "built by whoever processes them" if plan.recipes else format_bytes(total)
    here = (f"this computer uses {coordinator.local_machine.cores} cores" if coordinator.local_machine
            else "this computer only coordinates")
    say(f"{plan.command}: {len(plan.tiles)} tiles to process{which}{done}, {size}; {here}"
        + ("; helpers welcome" if helpers else ""))
    with _StopOnSigterm():
        coordinator.start()
        if helpers:
            say(f"Helpers join by running `zeit worker` (port {coordinator.port}).")
        result = coordinator.wait()
    say(summary(result))
    return result


def fetch_tiles(inputs, *, cache_dir: Optional[Union[str, Path]] = None, threads: int = 4,
                verbose: bool = True) -> List[Path]:
    """Download or build tiles described by recipes (``tile_urls``, ``stac_tiles``,
    ``gee_tiles``, or a ``.txt`` file of URLs) into the tile cache, where ``run_tiles`` looks for
    them before fetching them itself: on a cluster, run it on the login node, which has internet
    access, and the compute nodes process the tiles without it (with ``ZEIT_CACHE_DIR`` set to
    a folder both see). The same as ``zeit fetch tiles``. Returns the cached files."""
    from concurrent.futures import ThreadPoolExecutor

    from ._sources import recipe_key, run_build
    from ._worker import default_cache_dir

    sources = url_list(inputs)
    if not isinstance(sources, TileSources):
        raise ValueError("fetch_tiles takes tiles described by recipes (tile_urls, stac_tiles, gee_tiles, or a .txt "
                         "file of URLs); a folder of tiles is already here")
    cache = Path(cache_dir) if cache_dir else default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    say = (lambda text: print(text, flush=True)) if verbose else (lambda text: None)
    tiles = list(sources)

    def one(item):
        name, recipe = item
        target = cache / f"{recipe_key(recipe)}.tif"
        if target.is_file():
            return name, target, "already in the cache"
        part = cache / f"{target.stem}.fetch-{os.getpid()}.build.tif"
        log = cache / f"{target.stem}.fetch.log"
        try:
            ok, seconds, message = run_build(recipe, part, 1, log_path=log)
            if not ok:
                return name, None, message
            os.replace(part, target)
        finally:
            if part.exists():
                part.unlink()
            log.unlink(missing_ok=True)
        return name, target, f"{format_bytes(target.stat().st_size)} in {seconds:.0f} s"

    done: List[Path] = []
    failed: List[str] = []
    with ThreadPoolExecutor(max(1, threads)) as pool:
        for i, (name, target, how) in enumerate(pool.map(one, tiles), start=1):
            say(f"[{i}/{len(tiles)}] {name}: {how}")
            if target is None:
                failed.append(f"{name}: {how}")
            else:
                done.append(target)
    say(f"{len(done)} tiles in {cache}")
    if failed:
        raise RuntimeError(f"{len(failed)} of {len(tiles)} tiles could not be fetched (run it again: "
                           "the others are already in the cache):\n" + "\n".join(failed))
    return done


class _StopOnSigterm:
    """SIGTERM (a scheduler's warning before a job's time limit, or its cancellation) stops the
    run as Ctrl+C does, as ``Stopped``: the tiles being processed are dropped (their processes
    killed, their folders deleted) and the finished ones kept. Only in the main thread, where
    Python runs signal handlers."""

    def __enter__(self):
        self._previous = None
        if threading.current_thread() is threading.main_thread():
            def stop(signum, frame):
                raise Stopped(signal.Signals(signum).name)

            self._previous = signal.signal(signal.SIGTERM, stop)
        return self

    def __exit__(self, *exc):
        if self._previous is not None:
            signal.signal(signal.SIGTERM, self._previous)
        return False


def run_worker(*, cores: Optional[int] = None, connect: Optional[str] = None, join: Optional[str] = None,
               maps: Iterable[str] = (), cache_dir: Optional[str] = None, cache_gb: float = 20.0) -> None:
    """Lend this computer to the group: wait for tiles and process them, until Ctrl+C.

    Parameters
    ----------
    cores
        Cores to lend (default: all but one).
    connect
        The coordinator's address (``"192.168.1.10"`` or ``"host:port"``), when the
        broadcasts of the local network do not reach it.
    join
        The invitation code shown by ``zeit helpers invite`` (or ``zeit.invite_helpers``),
        the first time only.
    maps
        Where shared folders are mounted here, as ``"SRC=DST"``: a tile the coordinator sees
        under SRC is read here under DST (``"Z:\\=/mnt/nas"``).
    cache_dir, cache_gb
        Where received tiles are kept, and how much space they may take: a run with other
        options does not send them again.
    """
    from ._worker import run_worker as _run

    _run(cores=cores, connect=connect, join=join, maps=list(maps), cache_dir=cache_dir, cache_gb=cache_gb)


def invite_helpers(minutes: float = 10.0, *, port: int = _group.TCP_PORT) -> List[str]:
    """Show an invitation code and let other computers join this one's group with
    ``zeit worker --join <code>``, for ``minutes``. Returns the names of those that joined."""
    from ._pairing import Invitation

    invitation = Invitation(minutes=minutes, port=port)
    print(f"Invitation code: {invitation.code}  (valid for {minutes:g} minutes)\n"
          f"On each computer that will help, run:  zeit worker --join {invitation.code}", flush=True)
    try:
        return invitation.wait()
    except KeyboardInterrupt:
        return list(invitation.joined)
