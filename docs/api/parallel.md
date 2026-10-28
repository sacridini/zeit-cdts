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
    command, inputs, output_dir, *, helpers=False, local=True,
    local_cores=None, port=47801, compress_transfers=True,
    verbose=True, shard=None, **options,
)
```

Runs a command of the [zeit command line](../cli.md) on every tile, writing `<output_dir>/<tile name>/` with the command's outputs. Each tile runs in its own process, on this computer or on a helper, with all the cores it was given (the algorithms' threads). The largest tiles go first.

- **Resumable.** A finished tile is moved into place in one step, with a record of the options. Running the same call again skips the tiles already done, so an interrupted run continues where it stopped; other options for the same output folder raise an error rather than mixing results.
- **Helpers pull.** Each helper asks for a tile when it has room and fetches the next one while it processes the current one, so fast and slow machines balance by themselves. It reads a tile straight from shared storage (a NAS) when it sees the same file there, otherwise this computer sends it (compressed, when the file is not); tiles it already has from an earlier run are not sent again.
- **Results.** A helper that sees the output folder too (a NAS) writes its results straight into it; otherwise it sends them back.
- **No waiting for the slowest.** Near the end, an idle machine that would finish a straggler clearly sooner runs it too, and the first result wins. The last tile stays on this computer when it would finish it before a helper.
- **Failures.** A tile that fails is tried again on another machine; after two failures it is given up and reported, and the other tiles go on. A helper that stops responding gives its tiles back to the queue.
- **Warnings.** It says when the tiles are float32 without compression (they cross the network at full size), when the network is the bottleneck (the helpers wait for tiles more than they process them), and how to make helpers appear when none has joined.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `command` | `str` | required | A command that reads one raster: `"landtrendr"`, `"ccdc"`, `"bfast_monitor"`, `"bfast_lite"`, `"bfast"`, `"mann_kendall"`, `"phenology"`, `"smooth"`, `"tmask"`, `"twdtw"`, `"snic"`, `"classify"`, `"som"`, `"coded"`, `"bandpass_adjust"`. |
| `inputs` | path, glob, list or `TileSources` | required | A folder (its `.tif` files), a glob pattern (`"tiles/LT_*.tif"`), a list of files, a `.txt` file of URLs (one per line), or tiles each computer fetches or builds for itself: [`tile_urls`](#tile_urls), [`stac_tiles`](#stac_tiles), [`gee_tiles`](#gee_tiles). Two tiles cannot have the same name. |
| `output_dir` | path | required | Where the per-tile folders are written. |
| `helpers` | `bool` | `False` | Also send tiles to the computers of this computer's group running `zeit worker`. |
| `local` | `bool` | `True` | Process tiles here too. `False` (with `helpers=True`): only hand out tiles and collect the results, leaving this computer free. |
| `local_cores` | `int` | `None` | Cores for the tiles processed here; `None`: as [`n_jobs=-1`](#threads-n_jobs), all but one, or all of a cluster job's. |
| `port` | `int` | `47801` | TCP port the helpers connect to. |
| `compress_transfers` | `bool` | `True` | Send a tile stored without compression as a lossless ZSTD copy: the same values, fewer bytes on the network (1.43× fewer for Landsat NBR in float32, made at ~300 MB/s). |
| `verbose` | `bool` | `True` | Print the progress, the helpers that join or leave, and the warnings. |
| `shard` | `str` | `None` | `"i/n"`: process part `i` (from 0) of `n` of the tiles, tile `k` (in the order of `inputs`) being part `k mod n`'s. Inside a Slurm job array, `None` takes the task's part (see [Clusters](#clusters)). |
| `**options` | | | The command's options as keyword arguments, named as in the CLI with underscores: `max_segments=6`, `event_type="gain"`, `monitor_start_time="2020-01-01"`. A file used by every tile (`patterns=` of `twdtw`, `dates_file=` of `ccdc`, `model=` of `classify`, `training=` of `coded`) is sent to the helpers; a file that belongs to one raster (`weights=`, `samples=`) cannot be used. |

</div>

Returns a `TileRunResult`: `done` (the tiles processed now), `skipped` (already done before), `failed` (tile → error), `machines` (computer → tiles processed), `seconds`, and `ok` (no failures). On SIGTERM (a scheduler's warning before a job's time limit) it raises `zeit._tiles.Stopped`, a `KeyboardInterrupt` whose `done` lists the tiles finished before: the tiles being processed are dropped, the finished ones kept.

```python
result = zeit.run_tiles("bfast_monitor", "ndvi_16day/", "monitor/", helpers=True,
                        monitor_start_time="2022-01-01", h=0.25)
if not result.ok:
    print(result.failed)          # {"tile_07": "...the command's last lines of output..."}
```

## Tiles each computer builds

Instead of files on this computer, the tiles can be recipes: every computer then downloads or builds the tiles it processes, into its own cache (kept for the next run), and nothing passes through this computer. A recipe is data, a kind and its options checked against a fixed list, never code. It pays off when the computers have **independent internet links**: helpers on this computer's local network most likely share its link, and the run says so.

```python
tiles = zeit.stac_tiles(["21LYG", "21LZG", "22LBM"], index="NBR", bands=["nir08", "swir22"],
                        start_year=1985, end_year=2024, source="planetary_computer")
zeit.run_tiles("landtrendr", tiles, "lt_out/", helpers=True)
```

By default an index is written as int16, the index × 10000 with NoData −32768: what LandTrendr reads directly, and a third of the bytes of float32 once compressed.

### `tile_urls` { .api }

<!-- sig: zeit.tile_urls -->
```python
zeit.tile_urls(urls)
```

Rasters each computer downloads for itself: `https://` (a plain download) or a bucket GDAL reads (`s3://`, `gs://`, `az://`, with the usual credentials in the environment). Only URLs: a recipe never names a file of the computer that builds it.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `urls` | list or `dict` | required | URLs, each tile named after its file, or `{name: url}`. |

</div>

### `stac_tiles` { .api }

<!-- sig: zeit.stac_tiles -->
```python
zeit.stac_tiles(
    tiles, *, index=None, scale=10000.0, dtype="int16", **options,
)
```

Annual composites from a STAC catalog, built by each computer for the tiles it processes with [`build_annual_composites`](data.md#build_annual_composites), optionally reduced to one index with [`compute_indices`](data.md#compute_indices).

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `tiles` | list or `dict` | required | Sentinel-2 tile ids (`"23KPQ"`), `(min_lon, min_lat, max_lon, max_lat)` boxes, vector or raster files (their extent), or `{name: one of these}`. Each becomes a box in lon/lat. |
| `index` | `str` | `None` | `"NBR"`, `"NDVI"`, ...: one index instead of the bands. |
| `scale` | `float` | `10000.0` | With an integer `dtype`, values are written as `value * scale`. |
| `dtype` | `str` | `"int16"` | `"int16"`, `"int32"` (NoData: the type's minimum) or `"float32"` (values as they are). |
| `**options` | | | Of `build_annual_composites`: `source`, `collection`, `start_year`, `end_year`, `season`, `bands` (required), `cloud_cover_max`, `method`, `apply_cloud_mask`, `resolution`, `epsg`, `chunksize`. Without `epsg`, each tile is built in its own UTM zone, so `resolution` (30 by default) is in metres. |

</div>

### `gee_tiles` { .api }

<!-- sig: zeit.gee_tiles -->
```python
zeit.gee_tiles(
    tiles, *, start_year, end_year, index="NBR", project=None,
    method="auto", scale=10000.0, dtype="int16",
)
```

Annual medoid composites from Earth Engine, downloaded by each computer for the tiles it processes ([`download_gee_timeseries`](data.md#download_gee_timeseries) with `composite_type="annual"`) and stacked into one file per tile. Every computer needs Earth Engine credentials (`earthengine authenticate`). Earth Engine limits concurrent requests per project, so computers using the same project share that limit.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `tiles` | list or `dict` | required | Landsat WRS-2 ids (`"217/076"`, named `217_076`), Sentinel-2 ids, boxes, or `{name: tile}`. |
| `start_year`, `end_year` | `int` | required | The years, inclusive. |
| `index` | `str` | `"NBR"` | One index, or `None` for the six surface reflectance bands. |
| `project` | `str` | `None` | Google Cloud project for Earth Engine. |
| `method` | `str` | `"auto"` | Download method of `download_gee_image`: `"auto"`, `"direct"` or `"drive"`. |
| `scale`, `dtype` | | `10000.0`, `"int16"` | As in `stac_tiles`. |

</div>

## Lending a computer

### `run_worker` { .api }

<!-- sig: zeit.run_worker -->
```python
zeit.run_worker(
    *, cores=None, connect=None, join=None, maps=(), cache_dir=None,
    cache_gb=20.0,
)
```

Lends this computer to its group until Ctrl+C: it waits for a computer of the group that runs tiles, processes the tiles it gets, and waits again. The same as `zeit worker` in a terminal. A helper only opens outgoing connections and only runs zeit's own commands with the options it receives (never code sent by another computer), keeps the computer from sleeping while it works, and sets what the OpenMP runtime needs on macOS.

Both computers need the same zeit: a helper with another version (or other code, as in a development install) is told which version to install and gets no tiles.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `cores` | `int` | `None` | Cores to lend; `None`: as [`n_jobs=-1`](#threads-n_jobs). |
| `connect` | `str` | `None` | The coordinator's address, `"192.168.1.10"` or `"host:port"`, when the broadcasts of the local network do not reach it (another subnet, a VPN). |
| `join` | `str` | `None` | The invitation code shown by `invite_helpers` / `zeit helpers invite` on a computer of the group, the first time only. |
| `maps` | list of `str` | `()` | Where shared folders are mounted here, as `"SRC=DST"`: a tile the coordinator sees under SRC is read under DST, e.g. `"Z:\\=/mnt/nas"` or `"\\\\nas\\tiles=/Volumes/tiles"`. Mapped Windows drives are sent as their UNC paths, which Windows helpers read without a map. |
| `cache_dir` | path | `None` | Where received tiles are kept; `None`: the user's cache folder. |
| `cache_gb` | `float` | `20.0` | Space the kept tiles may take; the least recently used go first. |

</div>

### `invite_helpers` { .api }

<!-- sig: zeit.invite_helpers -->
```python
zeit.invite_helpers(minutes=10.0, *, port=47801)
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

## Clusters

On a cluster the tiles go to the tasks of a job array rather than to helpers (whose broadcasts a cluster's network does not carry; `helpers=True` inside a job is refused). Every task runs the same command, and takes its part of the tiles by itself:

```bash
#SBATCH --array=0-49 --cpus-per-task=16
zeit landtrendr tiles/ out/ --tiles      # task i processes tiles i, i + 50, i + 100, ...
```

- **The split.** Tile `k`, in the order of the inputs, goes to part `k mod n`, `n` being the number of tasks. Static, so it needs no lock on the shared file system (NFS and Lustre differ on them). The first task keeps the split in `out/.zeit-tiles/split.json`, so that tasks submitted again later (`--array=3,17,41`) take the same tiles. Other schedulers: `shard="i/n"` (`--shard`), e.g. `--shard $PBS_ARRAY_INDEX/50`.
- **Shared output folder.** Each run keeps its unfinished work apart (`out/.zeit-partial/<job>_<task>`), and a finished tile moves into place atomically.
- **The record.** `out/.zeit-tiles/` keeps the tiles of the run, what each task finished, what failed (with the error, and the tile's log in `logs/`) and whether it was stopped: [`zeit tiles status`](../cli.md#22-the-tiles-of-an-output-folder-zeit-tiles-status) reads it.
- **Time limits.** On SIGTERM (`#SBATCH --signal=B:TERM@120`, two minutes before the limit) the tile at work is dropped, its folder deleted, and with `--requeue` the task requeues itself (`scontrol requeue`) to continue with the tiles left.

[`zeit slurm`](../cli.md#21-a-job-array-for-the-tiles-zeit-slurm) writes the job script; the tutorial is [Zeit on a cluster (HPC, Slurm)](../tutorials/hpc-slurm.md).

## Threads (`n_jobs`)

Every function with `n_jobs` counts the threads of `n_jobs=-1` (the default) with one rule, the same in Python and in the compiled core. A positive `n_jobs` is the count. Otherwise, the first that applies:

1. `ZEIT_NUM_THREADS`, as it is. Tile runs set it for each tile's process.
2. In a worker of `dask.distributed`, the worker's threads (`--nthreads`, `threads_per_worker`), and not the `OMP_NUM_THREADS=1` its nanny sets for the BLAS libraries.
3. [`zeit config threads N`](../cli.md#24-this-users-settings-zeit-config), the user's own setting, outside a batch job.
4. `OMP_NUM_THREADS`, as it is.
5. The job's processors from its scheduler's variables, when the processors this process may run on are more (a cluster that does not confine its jobs): `SLURM_CPUS_PER_TASK`, or without `--cpus-per-task` `SLURM_CPUS_ON_NODE` and `SLURM_JOB_CPUS_PER_NODE`; `NCPUS` (PBS Pro) and `PBS_NUM_PPN` (Torque); `LSB_DJOB_NUMPROC` (LSF).
6. The processors this process may run on: its affinity (`taskset`, `srun --cpu-bind`) and the cgroup quota a scheduler or a container sets. In a job of Slurm, PBS or LSF (`SLURM_JOB_ID`, `PBS_JOBID`, `LSB_JOBID`), or when the process is confined to fewer than the machine has, all of them are used: they were given to it. Otherwise they are shared with whoever else is logged in to the machine (below), and on a computer of one's own one is kept free, so it stays responsive.

**A machine several people use at once.** On a Windows server reached by Remote Desktop, a Linux server reached by SSH, or a cluster's login node, `-1` divides the processors by the distinct users with an active session: 48 processors and 3 people logged in give 16 threads each, and the same machine at night, with one person, gives 47. Zeit counts users, not sessions (the same user in two windows counts once), and only active ones: on Windows the sessions `quser` lists as *Active* (a Remote Desktop closed without signing out is *Disconnected* and does not count), on Linux the logins of utmp (what `who` lists) or, without it, systemd-logind's sessions. Neither needs an administrator. The count is read again every 30 seconds. On the login node of a cluster (Slurm, PBS or LSF installed, outside a job), at most 4 threads: it is everyone's, and meant for editing and submitting jobs. Zeit warns once per process when it uses fewer threads for these reasons, and `zeit doctor` shows the users it counted; the results are the same, only the speed changes. `n_jobs`, `ZEIT_NUM_THREADS` or `zeit config threads N` set the count instead, and the last needs no environment variable (on Windows, a variable for the whole system needs an administrator).

**With Dask, leave `n_jobs` alone.** The blocks of a cube that run at once share the core's threads: there is one pool per thread count in a process, so four blocks with `n_jobs=-1` on dask's threads scheduler still run on as many threads as the processors, not four times as many, and dividing them among the blocks only leaves processors idle (LandTrendr on 2048 × 2048 pixels of 40 years, blocks of 512, 20 processors: 4.3 s with `-1`, 10.1 s with 5 threads per block). In a `distributed` worker, `-1` is the worker's threads (4 processes of 5 threads: 4.3 s, against 6.0 s with the nanny's single thread per block). One catch: creating a `LocalCluster` leaves `OMP_NUM_THREADS=1` in the process that created it, so zeit run there outside Dask then uses one thread; `ZEIT_NUM_THREADS` or `n_jobs` sets it back.

On a node of 128 cores, a job of `--cpus-per-task=16` gets 16 threads, without `--jobs`. Clusters' modules often export `OMP_NUM_THREADS=1` (for MPI programs): zeit then warns once that it leaves the job's other processors idle, and `ZEIT_NUM_THREADS`, read first, sets zeit's count without touching the other programs'.

### `doctor` { .api }

<!-- sig: zeit.doctor -->
```python
zeit.doctor(network=True)
```

Prints what zeit finds on this computer: the threads of `n_jobs=-1` and where the number comes from (`16 threads (16 of 128 processors available to this process; Slurm job 123456)`), the users with a session it divides them among, `zeit config threads` if set, the job and array task it runs in, the [caches](../getting-started/installation.md#caches) and whether they can be written, whether the services zeit downloads from answer (STAC, Earth Engine, Hugging Face, the embeddings), the Earth Engine credentials and the versions. The same as `zeit doctor` in a terminal: the first thing to run on a new cluster, and to paste in an issue.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `network` | `bool` | `True` | Check the services (a few seconds without internet access). |

</div>

### `fetch_tiles` { .api }

<!-- sig: zeit.fetch_tiles -->
```python
zeit.fetch_tiles(inputs, *, cache_dir=None, threads=4, verbose=True)
```

Downloads or builds the tiles of `inputs` (`tile_urls`, `stac_tiles`, `gee_tiles`, or a `.txt` file of URLs) into the tile cache, where `run_tiles` looks for them before fetching them itself. On a cluster, run it on the login node, which has internet access, with `ZEIT_CACHE_DIR` set to a folder the compute nodes also see: they then process the tiles without it. The same as `zeit fetch tiles`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `inputs` | recipes or path | | The tiles. A folder of tiles is already on disk, so it is refused. |
| `cache_dir` | path | `None` | Where to keep them (default: the `tiles` folder of the cache). |
| `threads` | `int` | `4` | Tiles fetched at once. |
| `verbose` | `bool` | `True` | Print each tile as it arrives. |

</div>

**Returns** the cached files. A tile that cannot be fetched does not stop the others: a `RuntimeError` at the end lists those that failed.

### `fetch_weights` { .api }

<!-- sig: zeit.fetch_weights -->
```python
zeit.fetch_weights(*names)
```

Downloads the pretrained weights of `zeit.ai` into the cache, checked by SHA-256, without importing torch: `"presto"`, `"prithvi-tiny-tl"`, `"prithvi-100m-tl"`, `"prithvi-300m"`, `"prithvi-300m-tl"`, `"prithvi-600m"`, `"prithvi-600m-tl"`, or `"all"` (also with no names). **Returns** the files. The same as `zeit fetch weights`.

