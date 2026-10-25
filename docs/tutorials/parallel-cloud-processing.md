# Parallel & Cloud Processing

<p class="lead">The same Zeit code runs on one laptop core, on every core of a workstation, or on several machines. This page explains the levels of parallelism, how to spread a folder of tiles over the computers of a local network with one command on each, how to set up a Dask cluster for a single cube larger than one machine, and how to write results that many machines can produce at once.</p>

<div class="glance" markdown>
<div><span class="k">Level 1</span><span class="v">Threads of the compiled core inside one process: <code>n_jobs</code></span></div>
<div><span class="k">Many tiles</span><span class="v">A folder of tiles here and on helpers: <code>--helpers</code>, <code>zeit worker</code></span></div>
<div><span class="k">Level 2</span><span class="v">Dask chunks across processes and machines: the <code>.zeit</code> accessor</span></div>
<div><span class="k">Storage</span><span class="v">Zarr for parallel writes, GeoTIFF for final products</span></div>
<div><span class="k">Rule</span><span class="v">Chunk in space, never in time</span></div>
</div>

## Levels of parallelism

**Threads within one machine.** Every algorithm's per-pixel loop runs in Rust, on several threads. `n_jobs=-1` (the default) uses all cores but one (in a cluster job, all of the job's), and needs nothing else: `zeit.landtrendr`, `zeit.ccdc` and friends are already parallel.

**Tiles across machines.** When the data already comes in tiles, each tile is a job of its own: `zeit landtrendr tiles/ out/ --helpers` runs them here and on the computers of the local network running `zeit worker`. See [Many tiles on several computers](#many-tiles-on-several-computers): it is the simplest way to use more machines.

**Chunks across processes or machines.** For a single cube larger than memory, or one area split over more machines, wrap the data in a Dask-backed xarray cube and pass it to the algorithm (`zeit.landtrendr(cube)`, `zeit.mann_kendall(cube)`, …) or to its `.zeit` accessor form. Zeit maps the compiled code over spatial chunks, and Dask schedules those chunks on its workers.

```python
import xarray as xr
import zeit   # registers the .zeit accessor

cube = xr.open_zarr("s3://my-bucket/annual_ndvi.zarr")["ndvi"]        # (time, y, x), lazy
cube = cube.chunk({"time": -1, "y": 512, "x": 512})                   # whole history per chunk

trend = cube.zeit.mann_kendall(method="hamed_rao", n_jobs=1)          # still lazy: a Dataset of maps
trend.to_zarr("s3://my-bucket/ndvi_trend.zarr", consolidated=True)    # computes, in parallel
```

!!! warning "Chunk in space, never in time"
    Every algorithm needs a pixel's complete history, so the `time` axis must be a **single chunk** (`-1`). Chunk only `y` and `x`. Typical sizes are 256–1024 pixels, small enough that a chunk's full history fits in a worker's memory.

!!! tip "Avoid oversubscription"
    If Dask already runs one task per core, give each task one thread (`n_jobs=1`). If instead you run **one worker process per machine**, let that worker use all its cores (`n_jobs=-1`, `--nthreads 1`). Mixing both gives cores × cores threads fighting over the CPU.

## Many tiles on several computers

When the work is a folder of tiles (Landsat tiles for LandTrendr, say), every tile is a job of its own and no cluster is needed. Zeit runs the folder on this computer, and any computer of the local network can lend itself with one command. No scheduler, no ports to choose, no versions to align by hand.

### Once: bring the computers into a group

On the computer that has the tiles:

```bash
zeit helpers invite              # shows a code, e.g. K7QM-2XDP, valid for 10 minutes
zeit helpers firewall            # Windows only: lets the helpers reach this computer (asks for admin rights)
```

On each computer that will help, with the same zeit version installed:

```bash
zeit worker --join K7QM-2XDP     # joins the group, then waits for work
```

From then on, `zeit worker` is all a helper needs (in a terminal it can leave open, or at startup). It lends all its cores but one (`--cores` to lend fewer), only runs zeit's own commands, and keeps the computer from sleeping while it works. Any computer of the group can run tiles with the others' help, and invite more.

### Running the tiles

```bash
zeit landtrendr nbr_tiles/ lt_out/ --helpers --max-segments 6              # here and on the helpers
zeit landtrendr nbr_tiles/ lt_out/ --helpers --no-local --max-segments 6   # only on the helpers
```

```python
import zeit

result = zeit.run_tiles("landtrendr", "nbr_tiles/", "lt_out/", helpers=True, max_segments=6)
result = zeit.run_tiles("landtrendr", "nbr_tiles/", "lt_out/", helpers=True, local=False)   # only coordinate
```

Every tile gets its own folder, `lt_out/<tile name>/`, with what `zeit landtrendr` writes for one file. With `--no-local` (`local=False`) this computer processes nothing: it hands out the tiles and collects the results, and stays free for other work. It must stay on, though: the results come back to it.

The run prints the helpers as they join and one progress line:

```text
landtrendr: 120 tiles to process, 538.2 GB; this computer uses 19 cores; helpers welcome
Helpers join by running `zeit worker` (port 47801).
+ lab-03 (192.168.1.23, 15 cores) joined
+ lab-07 (192.168.1.27, 7 cores) joined
landtrendr: 37/120 tiles | 3 computers | sending 104.3 MB/s | ~1 h 12 min left
```

What happens underneath:

- **The helpers pull.** Each asks for a tile when it has room and fetches the next while it processes the current one, so a fast workstation takes more tiles than a laptop without any setting. The largest tiles go first.
- **No waiting for the slowest.** Near the end, an idle computer that would finish a straggler clearly sooner runs it too, and the first result wins. The last tile stays on the main computer when it would finish it before a helper.
- **Nothing is lost.** A tile that fails is tried again on another computer (after two failures it is listed at the end). A helper that is switched off gives its tiles back to the queue. Stopping the run (Ctrl+C) keeps the finished tiles, and running the same command again processes only the others.

### Where the tiles come from

- **The main computer's disk.** The main computer sends each tile to the helper that gets it, in 16 MB pieces read one at a time, so a hard disk is never asked for several files at once (which is much slower than reading them in turn).
- **A NAS that every computer sees.** A helper that sees the same file reads it from the NAS itself, and the main computer sends nothing. Mapped Windows drives travel as their UNC path (`\\nas\tiles\...`), which Windows helpers read directly; a Linux or macOS helper says where the share is mounted: `zeit worker --map "\\nas\tiles=/mnt/tiles"`. The helper copies the whole tile before processing it, which is much faster than GDAL reading it piece by piece over the network.
- **Again.** A helper keeps the tiles it received (20 GB by default, `--cache-gb`), so a second run with other options does not send them again.
- **The results** go back the same way: a helper that sees the output folder too (on a NAS) writes them straight into it, otherwise it sends them to the main computer. They are small next to the tiles.

### What limits the speed: the network

A full Landsat tile, 54.7 million pixels × 41 years, takes about **98 s** of LandTrendr on a 20-thread desktop (see [Performance](../benchmarks/performance.md)). Sending that tile to a helper over a 1 Gbit/s cable (about 110 MB/s) takes:

| Tile | Size | Sent in | Helpers the main computer's cable keeps busy* |
| :--- | :---: | :---: | :---: |
| float32, no compression | 9.0 GB | ~80 s | ~1 |
| float32, as Zeit sends it (lossless ZSTD) | ~6.3 GB | ~57 s | ~1.7 |
| int16 (NBR × 10000), no compression | 4.5 GB | ~40 s | ~2.5 |
| int16 (NBR × 10000), ZSTD compression | ~2.9 GB | ~26 s | ~4 |
| the same over Wi-Fi | ~2.9 GB | ~60–100 s | ~1–2 |

\* As fast as that desktop: the time to process a tile divided by the time to send it. Slower helpers take longer per tile, so more of them fit. The compressed sizes were measured on the NBR of that tile (215/067): ZSTD with the floating-point predictor makes float32 1.43× smaller, int16 with ZSTD is 3.1× smaller than float32. Other data compresses differently.

So the format of the tiles matters more than the number of helpers. An index scaled by 10000 as int16 (as Earth Engine exports NBR) with ZSTD compression is a fraction of a float32 stack; the conversion costs four decimal places, so it is your call, not Zeit's. A NAS changes where the bottleneck is (its own network link and disks) but not the arithmetic. A tile stored without compression is sent as a lossless ZSTD copy (made at ~300 MB/s with two threads, faster than a 1 Gbit/s link carries it; `compress_transfers=False` turns it off). The run warns when the tiles are float32 without compression, and when the helpers spend more than a quarter of their time waiting for tiles: then more helpers will not make it faster.

### Tiles each computer downloads for itself

The tiles do not have to exist yet. Given as recipes, each computer downloads or builds the tiles it processes, into its own cache, and nothing passes through the main computer:

```python
zeit.run_tiles("landtrendr", zeit.tile_urls(["https://data.example.org/nbr/215_067.tif", ...]), "lt_out/",
               helpers=True)                                               # files on a web server or a bucket

tiles = zeit.stac_tiles(["21LYG", "21LZG"], index="NBR", bands=["nir08", "swir22"],
                        start_year=1985, end_year=2024)                    # annual composites from STAC
tiles = zeit.gee_tiles(["215/067", "216/067"], start_year=1985, end_year=2024)   # medoids from Earth Engine
zeit.run_tiles("landtrendr", tiles, "lt_out/", helpers=True)
```

From the terminal, a text file of URLs (one per line) works as the input: `zeit landtrendr urls.txt lt_out/ --helpers`. The composites are written as int16 (the index × 10000) unless `dtype="float32"`.

This only pays off when the computers have **independent internet links** (another building, a colleague's home over a VPN, cloud machines near the data). In the same office they share one link: a Landsat tile built from STAC is ~41 GB of scenes, so the link, not the number of computers, sets the pace. Earth Engine limits concurrent requests per project, not per computer. The run warns when a helper is on the main computer's network.

### When the helpers do not appear

- **Windows.** Run `zeit helpers firewall` on the main computer (the helpers need no rule: they only connect out). A network marked *Public* blocks incoming connections without it.
- **Another subnet or a VPN.** Broadcasts do not cross routers. Start the helper with the main computer's address: `zeit worker --connect 192.168.2.10`.
- **Which helpers are listening.** `zeit helpers list` on any computer of the group shows the helpers waiting for work, with their cores and zeit version.
- **Another zeit version.** A helper with another version (or other code, as with a development install) is told which version to install, and the main computer says so too.

## Setting up a cluster

For a single cube that does not fit one machine. A Dask cluster has one **scheduler**, which hands out work, and any number of **workers**, which do it.

### On one machine

```python
from dask.distributed import Client, LocalCluster

client = Client(LocalCluster(n_workers=4, threads_per_worker=1))
print(client.dashboard_link)   # live view of tasks, memory and CPU
```

### Across machines on a local network

On the machine that will coordinate:

```bash
dask scheduler                 # prints its address, e.g. tcp://192.168.1.10:8786
```

On every other machine (with Zeit installed in the same Python version):

```bash
dask worker tcp://192.168.1.10:8786 --nworkers 4 --nthreads 1 --memory-limit 8GB
```

Then connect from your script:

```python
from dask.distributed import Client
client = Client("tcp://192.168.1.10:8786")
```

If a worker cannot connect or crashes, see [Troubleshooting](#troubleshooting-a-dask-cluster) below.

### In the cloud

Dask has launchers for most platforms: `dask-kubernetes` (Kubernetes), `dask-cloudprovider` (AWS, GCP, Azure), `dask-jobqueue` (SLURM, PBS on HPC systems), or managed services such as Coiled.

```python
from dask_kubernetes.operator import KubeCluster
from dask.distributed import Client

cluster = KubeCluster(name="zeit", image="ghcr.io/dask/dask:latest", n_workers=20)
client = Client(cluster)
```

Workers need Zeit installed: use the [Zeit Docker image](../getting-started/docker.md) or add `pip install zeit-cdts` to the worker image.

## Writing results: Zarr

When many workers write at once, a single GeoTIFF becomes a bottleneck (or gets corrupted). **Zarr** stores an array as a folder of independently compressed chunks, so every worker writes its own piece, locally or straight to S3 / Google Cloud Storage.

`.zeit.to_zarr_optimized()` rechunks the result (512 × 512 by default), writes it and consolidates the metadata so it reads fast from object storage:

```python
result.zeit.to_zarr_optimized("gs://my-bucket/result.zarr")
```

!!! tip "Don't `.compute()` a large result"
    `.compute()` pulls the whole array into the memory of your client. Write it with `to_zarr_optimized` (or `save_raster` for results that fit in memory) and let the workers stream it to storage.

## A complete example

LandTrendr over a large annual NDVI cube stored as Zarr, on a cluster:

```python
import xarray as xr
from dask.distributed import Client
import zeit

client = Client("tcp://192.168.1.10:8786")

ndvi = xr.open_zarr("gs://my-bucket/annual_ndvi.zarr")["ndvi"]      # (time, y, x), 1985-2024
ndvi = ndvi.chunk({"time": -1, "y": 512, "x": 512})

# Lazy: the years come from the time coordinate; direction="loss" looks for NDVI drops.
lt = zeit.landtrendr(ndvi, max_segments=6, n_jobs=1)
lt.to_zarr("gs://my-bucket/landtrendr_vertices.zarr", mode="w")     # computes, in parallel

# Event maps from the saved vertices, again chunk by chunk
lt = xr.open_zarr("gs://my-bucket/landtrendr_vertices.zarr")
loss = zeit.extract_events(lt, min_magnitude=2000)
loss.to_zarr("gs://my-bucket/landtrendr_loss.zarr", mode="w")       # yod, magnitude, duration, ...
```

`zeit.landtrendr` and `extract_events` return lazy `xarray.Dataset`s here: each `to_zarr` runs the compiled code on every chunk in parallel and streams the result to storage. Saving the vertices first means they are computed once, and other event maps (`event_type="gain"`, another `sort_by`) come from them without segmenting again. The dashboard (`client.dashboard_link`, port 8787 by default) shows progress, memory and CPU for every worker.

## Troubleshooting a Dask cluster

Connecting a mixed-OS cluster (e.g. a Windows desktop as scheduler + a macOS laptop as a worker) over a home/office network hits a handful of predictable snags. Here's what to check, roughly in the order you'll hit them.

### Windows Firewall blocks the remote worker

Windows often marks a home/office network as **Public**, which blocks unsolicited inbound connections by default — the scheduler will start fine locally, but a worker on another machine will simply never be able to reach it.

Fix it with a firewall rule scoped to your LAN subnet (run in an **elevated** PowerShell), rather than opening the ports to the world:

```powershell
New-NetFirewallRule -DisplayName "Dask Scheduler (LAN)" -Direction Inbound -Protocol TCP `
    -LocalPort 8786,8787 -Action Allow -RemoteAddress 192.168.1.0/24
```

Replace `192.168.1.0/24` with your actual subnet. `8786` is the scheduler port, `8787` the dashboard.

### Keep Python, dask and distributed versions aligned on every machine

The scheduler/worker wire protocol assumes matching (or very close) `dask`/`distributed` versions; a mismatch triggers a `VersionMismatchWarning` and can cause hard-to-diagnose failures under load. Before connecting a new worker, check:

```bash
python -c "import sys, dask, distributed; print(sys.version, dask.__version__, distributed.__version__)"
```

...and make sure it's close to what the scheduler machine reports. `zeit`'s published PyPI wheels currently cover Python 3.9–3.12 (Windows, Linux, macOS arm64) — there is no prebuilt 3.13 wheel yet, so standardize on a **Python 3.12** environment (venv or conda) on every machine to avoid an accidental from-source build.

### macOS: a worker crash-loops silently the moment a task touches `zeit`

**Symptom:** `dask worker` starts and registers with the scheduler fine. But as soon as a real task imports `zeit` (e.g. the first `zeit.landtrendr()` call), the worker process vanishes and Dask's Nanny silently respawns it with a new port — forever. No Python traceback reaches the scheduler or client; calling `client.run(...)` against that worker just raises `CommClosedError: ... Stream is closed`.

**Cause:** two copies of the OpenMP runtime (`libomp`/`libiomp`) in one process. Up to zeit 1.7, `torch` and Zeit's own compiled C++ extension (`zeit._core`) each linked one; from 2.0 Zeit's core is Rust and links none, so a second copy can only come from another package of the environment. Loading both inside the same process aborts the whole process (`OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized`) instead of raising a catchable Python exception — which is exactly what a Nanny-managed silent restart loop looks like from the outside.

**Fix:** set these two environment variables before launching the worker on macOS:

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 dask worker tcp://<scheduler-ip>:8786 --nworkers <n> --nthreads 1
```

If a worker is still crash-looping and you need to see the actual OS-level error, look at the raw terminal where `dask worker` runs directly — `Segmentation fault`, `Illegal instruction`, or the `OMP: Error #15` line only ever prints there, never through the Dask protocol.

### `dask worker` can't find `zeit` even though you just installed it

If `pip install zeit-cdts` (or `pip install -e .`) reported success but a worker still throws `ModuleNotFoundError: No module named 'zeit'`, the `dask` command on your `PATH` is almost certainly resolving to a *different* Python installation (a different conda env, a system Python, a pyenv shim) than the one you installed `zeit` into.

Force it explicitly — activate the right environment, then launch via `python -m dask` instead of the bare `dask` binary, so it always uses the currently active interpreter:

```bash
conda activate zeit-worker   # or: source your-venv/bin/activate
python -m dask worker tcp://<scheduler-ip>:8786 --nworkers <n> --nthreads 1
```

### Sanity-check every worker before submitting real work

From the client/head node, verify `zeit` actually imports on every connected worker *before* kicking off a real job — it's much faster to catch a broken worker this way than to debug a stuck/slow distributed run:

```python
from dask.distributed import Client

client = Client("tcp://<scheduler-ip>:8786")

def check():
    import socket, sys
    try:
        from zeit import landtrendr
        return (socket.gethostname(), sys.executable, "ok")
    except Exception as e:
        return (socket.gethostname(), sys.executable, repr(e))

print(client.run(check, on_error="return"))
```

A clean `"ok"` (or a normal, readable Python exception) per worker means you're good to go. A `CommClosedError` / `Stream is closed` here is the macOS crash-loop symptom described above ("macOS: a worker crash-loops") — fix that first.

### Too many local read threads can be *slower* than reading sequentially

If your input files only exist on the machine that's also acting as the client (the common case when the other machines don't share a network drive), you have to read them locally before handing chunks to the cluster. It's tempting to parallelize that read with a large thread pool, but past a certain point you're not adding throughput — you're adding disk-queue contention.

Measured reading 41 full-resolution GeoTIFFs (~218MB each, ~8.9GB total) from a local NVMe drive:

| Threads | Time |
|---|---|
| 1 (sequential) | 26.6s |
| 2 | 22.9s |
| **4** | **22.0s (best)** |
| 16 | 100.7s (4.6x *slower* than sequential!) |

A modest thread pool (3-4) beat both sequential and 16 threads by a wide margin. Benchmark this on your own disk before picking a number — the right count depends on whether it's NVMe, SATA SSD, spinning disk, or a network share, but "more threads" is not a safe default assumption.

### Gathering a big distributed result can OOM a single worker, even though the cluster computed it fine

If you call `.compute()` (or `client.gather()`) directly on a large persisted Dask array, Dask may run a "finalize" (concatenation) task on a single worker to assemble the pieces before handing them to the client — and that worker's own `--memory-limit` might be far smaller than the assembled result, even though every individual chunk fit in memory just fine:

```
MemoryError: Task 'finalize-...' has 2.84 GiB worth of input dependencies,
but worker tcp://192.168.2.40:55301 has memory_limit set to 1.60 GiB.
```

Avoid this by gathering block-by-block instead of triggering one big finalize, and assembling the final array yourself on the client (which usually has far more RAM to spare than a single worker's configured limit):

```python
import numpy as np

n_bands = persisted.shape[0]
y_chunks, x_chunks = persisted.chunks[1], persisted.chunks[2]
y_off = np.concatenate([[0], np.cumsum(y_chunks)])
x_off = np.concatenate([[0], np.cumsum(x_chunks)])

block_futs = {
    (by, bx): client.compute(persisted.blocks[:, by, bx])
    for by in range(len(y_chunks))
    for bx in range(len(x_chunks))
}
gathered = client.gather(block_futs)

result = np.empty((n_bands, H, W), dtype=np.float32)
for (by, bx), arr in gathered.items():
    result[:, y_off[by]:y_off[by + 1], x_off[bx]:x_off[bx + 1]] = arr
```

This also parallelizes the fetch itself (all blocks are requested up front, not one at a time), so it's usually *faster* than the naive `.compute()` in addition to being memory-safe.
