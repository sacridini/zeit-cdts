# Many Tiles & Helpers

<p class="lead">Run a command on every tile of a folder, on this computer and on other computers of the local network that lend themselves to it. No cluster to set up: one command on each helper, one on the computer that has the tiles. Tutorial: <a href="../../tutorials/parallel-cloud-processing/#many-tiles-on-several-computers">Many tiles on several computers</a>.</p>

```python
# On each computer that helps (once: zeit worker --join <code>; see invite_helpers)
zeit.run_worker()

# On the computer with the tiles
zeit.run_tiles("landtrendr", "nbr_tiles/", "lt_out/", helpers=True, max_segments=6)
zeit.run_tiles("landtrendr", "nbr_tiles/", "lt_out/", helpers=True, local=False)   # only coordinate
```

The same is available from the terminal: `zeit landtrendr nbr_tiles/ lt_out/ --helpers`, `zeit worker`, `zeit helpers invite` (see the [CLI](../cli.md#17-many-tiles-here-and-on-other-computers)).

## Running tiles

### `run_tiles` { .api }

<!-- sig: zeit.run_tiles -->
```python
zeit.run_tiles(
    command, inputs, output_dir, helpers=False, local=True,
    local_cores=None, port=47801, verbose=True, **options,
)
```

Runs a command of the [zeit command line](../cli.md) on every tile, writing `<output_dir>/<tile name>/` with the command's outputs. Each tile runs in its own process, on this computer or on a helper, with all the cores it was given (the algorithms' OpenMP threads). The largest tiles go first.

- **Resumable.** A finished tile is moved into place in one step, with a record of the options. Running the same call again skips the tiles already done, so an interrupted run continues where it stopped; other options for the same output folder raise an error rather than mixing results.
- **Helpers pull.** Each helper asks for a tile when it has room and fetches the next one while it processes the current one, so fast and slow machines balance by themselves. It reads a tile straight from shared storage (a NAS) when it sees the same file there, otherwise this computer sends it; tiles it already has from an earlier run are not sent again.
- **No waiting for the slowest.** Near the end, an idle machine that would finish a straggler clearly sooner runs it too, and the first result wins. The last tile stays on this computer when it would finish it before a helper.
- **Failures.** A tile that fails is tried again on another machine; after two failures it is given up and reported, and the other tiles go on. A helper that stops responding gives its tiles back to the queue.
- **Warnings.** It says when the tiles are float32 without compression (they cross the network at full size), when the network is the bottleneck (the helpers wait for tiles more than they process them), and how to make helpers appear when none has joined.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `command` | `str` | required | A command that reads one raster: `"landtrendr"`, `"ccdc"`, `"bfast_monitor"`, `"bfast_lite"`, `"bfast"`, `"mann_kendall"`, `"phenology"`, `"smooth"`, `"tmask"`, `"twdtw"`, `"snic"`, `"classify"`, `"som"`, `"coded"`, `"bandpass_adjust"`. |
| `inputs` | path, glob or list | required | A folder (its `.tif` files), a glob pattern (`"tiles/LT_*.tif"`) or a list of files. Two tiles cannot have the same name. |
| `output_dir` | path | required | Where the per-tile folders are written. |
| `helpers` | `bool` | `False` | Also send tiles to the computers of this computer's group running `zeit worker`. |
| `local` | `bool` | `True` | Process tiles here too. `False` (with `helpers=True`): only hand out tiles and collect the results, leaving this computer free. |
| `local_cores` | `int` | `None` | Cores for the tiles processed here; `None`: all but one. |
| `port` | `int` | `47801` | TCP port the helpers connect to. |
| `verbose` | `bool` | `True` | Print the progress, the helpers that join or leave, and the warnings. |
| `**options` | | | The command's options as keyword arguments, named as in the CLI with underscores: `max_segments=6`, `event_type="gain"`, `monitor_start_time="2020-01-01"`. A file used by every tile (`patterns=` of `twdtw`, `dates_file=` of `ccdc`, `model=` of `classify`, `training=` of `coded`) is sent to the helpers; a file that belongs to one raster (`weights=`, `samples=`) cannot be used. |

</div>

Returns a `TileRunResult`: `done` (the tiles processed now), `skipped` (already done before), `failed` (tile → error), `machines` (computer → tiles processed), `seconds`, and `ok` (no failures).

```python
result = zeit.run_tiles("bfast_monitor", "ndvi_16day/", "monitor/", helpers=True,
                        monitor_start_time="2022-01-01", h=0.25)
if not result.ok:
    print(result.failed)          # {"tile_07": "...the command's last lines of output..."}
```

## Lending a computer

### `run_worker` { .api }

<!-- sig: zeit.run_worker -->
```python
zeit.run_worker(
    cores=None, connect=None, join=None, maps=(), cache_dir=None,
    cache_gb=20.0,
)
```

Lends this computer to its group until Ctrl+C: it waits for a computer of the group that runs tiles, processes the tiles it gets, and waits again. The same as `zeit worker` in a terminal. A helper only opens outgoing connections and only runs zeit's own commands with the options it receives (never code sent by another computer), keeps the computer from sleeping while it works, and sets what the OpenMP runtime needs on macOS.

Both computers need the same zeit: a helper with another version (or other code, as in a development install) is told which version to install and gets no tiles.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `cores` | `int` | `None` | Cores to lend; `None`: all but one. |
| `connect` | `str` | `None` | The coordinator's address, `"192.168.1.10"` or `"host:port"`, when the broadcasts of the local network do not reach it (another subnet, a VPN). |
| `join` | `str` | `None` | The invitation code shown by `invite_helpers` / `zeit helpers invite` on a computer of the group, the first time only. |
| `maps` | list of `str` | `()` | Where shared folders are mounted here, as `"SRC=DST"`: a tile the coordinator sees under SRC is read under DST, e.g. `"Z:\\=/mnt/nas"` or `"\\\\nas\\tiles=/Volumes/tiles"`. Mapped Windows drives are sent as their UNC paths, which Windows helpers read without a map. |
| `cache_dir` | path | `None` | Where received tiles are kept; `None`: the user's cache folder. |
| `cache_gb` | `float` | `20.0` | Space the kept tiles may take; the least recently used go first. |

</div>

### `invite_helpers` { .api }

<!-- sig: zeit.invite_helpers -->
```python
zeit.invite_helpers(minutes=10.0, port=47801)
```

Shows an invitation code (such as `K7QM-2XDP`) and, for `minutes`, lets other computers join this computer's group with `zeit worker --join <code>`. The same as `zeit helpers invite`. Returns the names of the computers that joined. A computer joins once; afterwards `zeit worker` is enough. Any computer of the group can then run tiles with the others' help, and invite more.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `minutes` | `float` | `10.0` | How long the code is valid. |
| `port` | `int` | `47801` | TCP port of the invitation. |

</div>

!!! note "Security"
    The code is only an invitation: it authenticates one pairing (over TLS, with a key stretched by scrypt; ten wrong codes close the invitation), in which the newcomer receives the group's random 256-bit secret. Every later connection proves the secret both ways, bound to the coordinator's TLS certificate, so another machine of the network can neither pose as a member nor read the tiles. The secret is kept in the user's configuration folder (`%APPDATA%\zeit` on Windows, `~/.config/zeit` elsewhere); `zeit helpers leave` forgets it.
