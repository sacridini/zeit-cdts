"""Many tiles, on this computer and on helpers in the local network.

``zeit.run_tiles`` (or ``zeit <command> <folder> <output> --tiles``) runs a command on every
tile of a folder, here and on the computers running ``zeit worker`` in the same group.
Finished tiles are kept, so running it again continues where it stopped.
"""

from pathlib import Path
from typing import Callable, Iterable, List, Optional, Union

from . import _group
from ._coordinator import Coordinator, TileRunResult, summary
from ._tasks import format_bytes, make_plan, option_values

__all__ = ["run_tiles", "run_worker", "invite_helpers", "TileRunResult"]


def run_tiles(command: str, inputs: Union[str, Path, Iterable], output_dir: Union[str, Path], *,
              helpers: bool = False, local: bool = True, local_cores: Optional[int] = None,
              port: int = _group.TCP_PORT, verbose: bool = True, **options) -> TileRunResult:
    """Run ``command`` on every tile, writing ``<output_dir>/<tile name>/``.

    Parameters
    ----------
    command
        A command of the zeit command line that reads one raster: ``"landtrendr"``,
        ``"ccdc"``, ``"bfast_monitor"``, ``"mann_kendall"``, ``"phenology"``, ...
    inputs
        A folder (its ``.tif`` files), a glob pattern (``"tiles/LT_*.tif"``) or a list of files.
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
    **options
        The command's options, as keyword arguments (``max_segments=6``, ``event_type="gain"``).

    Returns
    -------
    TileRunResult
        The tiles done now, those skipped as already done, the failed ones (with the error)
        and how many each computer processed.

    Examples
    --------
    >>> zeit.run_tiles("landtrendr", "nbr_tiles/", "lt_out/", helpers=True, max_segments=6)
    >>> zeit.run_tiles("landtrendr", "nbr_tiles/", "lt_out/", helpers=True, local=False)   # only coordinate
    """
    out: Optional[Callable[[str], None]] = None if verbose else (lambda text: None)
    command = command.replace("_", "-")
    plan = make_plan(command, inputs, output_dir, option_values(command, options))
    say = out or (lambda text: print(text, flush=True))
    if not plan.tiles:
        say(f"All {len(plan.done)} tiles are already done in {plan.output_dir}.")
        return TileRunResult(plan.output_dir, [], [t.stem for t in plan.done], {}, {}, 0.0)
    coordinator = Coordinator(plan, local=local, local_cores=local_cores, helpers=helpers, port=port, out=out)
    total = sum(t.size for t in plan.tiles)
    done = f" ({len(plan.done)} already done)" if plan.done else ""
    here = (f"this computer uses {coordinator.local_machine.cores} cores" if coordinator.local_machine
            else "this computer only coordinates")
    say(f"{plan.command}: {len(plan.tiles)} tiles to process{done}, {format_bytes(total)}; {here}"
        + ("; helpers welcome" if helpers else ""))
    coordinator.start()
    if helpers:
        say(f"Helpers join by running `zeit worker` (port {coordinator.port}).")
    result = coordinator.wait()
    say(summary(result))
    return result


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
