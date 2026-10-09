# Parallel & Cloud Processing

<p class="lead">The same Zeit code runs on one laptop core, on every core of a workstation, or on a cluster of machines. This page explains the two levels of parallelism, how to set up a Dask cluster, and how to write results that many machines can produce at once.</p>

<div class="glance" markdown>
<div><span class="k">Level 1</span><span class="v">C++ threads (OpenMP) inside one process: <code>n_jobs</code></span></div>
<div><span class="k">Level 2</span><span class="v">Dask chunks across processes and machines: the <code>.zeit</code> accessor</span></div>
<div><span class="k">Storage</span><span class="v">Zarr for parallel writes, GeoTIFF for final products</span></div>
<div><span class="k">Rule</span><span class="v">Chunk in space, never in time</span></div>
</div>

## Two levels of parallelism

**Threads within one machine.** Every algorithm's per-pixel loop runs in C++ with OpenMP. `n_jobs=-1` (the default) uses all cores but one, and needs nothing else: `zeit.landtrendr`, `zeit.ccdc` and friends are already parallel.

**Chunks across processes or machines.** For data larger than memory, or more machines, wrap the data in a Dask-backed xarray cube and pass it to the algorithm (`zeit.landtrendr(cube)`, `zeit.mann_kendall(cube)`, …) or to its `.zeit` accessor form. Zeit maps the C++ code over spatial chunks, and Dask schedules those chunks on its workers.

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

## Setting up a cluster

A Dask cluster has one **scheduler**, which hands out work, and any number of **workers**, which do it.

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

If a worker cannot connect or crashes, see [Troubleshooting](#troubleshooting-a-multi-machine-cluster) below.

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

`zeit.landtrendr` and `extract_events` return lazy `xarray.Dataset`s here: each `to_zarr` runs the C++ code on every chunk in parallel and streams the result to storage. Saving the vertices first means they are computed once, and other event maps (`event_type="gain"`, another `sort_by`) come from them without segmenting again. The dashboard (`client.dashboard_link`, port 8787 by default) shows progress, memory and CPU for every worker.

## Troubleshooting a multi-machine cluster

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

**Cause:** `zeit` depends on `torch`, and on macOS both `torch` and Zeit's own compiled C++ extension (`zeit._core`) link their own copy of the OpenMP runtime (`libomp`/`libiomp`). Loading both inside the same process aborts the whole process (`OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized`) instead of raising a catchable Python exception — which is exactly what a Nanny-managed silent restart loop looks like from the outside.

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
