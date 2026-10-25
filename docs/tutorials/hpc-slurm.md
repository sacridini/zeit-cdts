# Zeit on a Cluster (HPC, Slurm)

<p class="lead">A cluster gives zeit many processors and many nodes, through a scheduler and with a few habits of its own: compute nodes without internet access, a small home directory, modules that set <code>OMP_NUM_THREADS=1</code>, time limits that stop a job midway. This page goes from installing zeit on a cluster to a job array that processes a thousand tiles, resubmits what failed and survives its time limit, with Slurm as the example.</p>

<div class="glance" markdown>
<div><span class="k">First</span><span class="v"><code>zeit doctor</code> in a job: threads, caches, network, what to fix</span></div>
<div><span class="k">One node</span><span class="v"><code>sbatch</code> with <code>--cpus-per-task</code>; zeit uses them all, without <code>--jobs</code></span></div>
<div><span class="k">Many nodes</span><span class="v"><code>zeit slurm</code> writes a job array over the tiles; <code>zeit tiles status</code> says what is left</span></div>
<div><span class="k">No internet</span><span class="v"><code>zeit fetch</code> on the login node, into a cache the nodes see</span></div>
<div><span class="k">Containers</span><span class="v">Apptainer, from the image published with each version</span></div>
</div>

## Installing

### In a virtual environment

On the login node, with the cluster's Python (often `module load python/3.12`):

```bash
python -m venv ~/venvs/zeit
source ~/venvs/zeit/bin/activate
pip install zeit-cdts            # the algorithms, the cubes, the tiles
pip install "zeit-cdts[ai]"      # and zeit.ai, with PyTorch (gigabytes: only if you use it)
```

Zeit is published as wheels for Linux x86_64 and aarch64 (Grace, Graviton), for glibc 2.28 or newer (RHEL 8, Rocky 8, Ubuntu 20.04 and later). A job runs zeit with the environment's Python; activating the environment in the job script is enough.

### Where the caches go

What zeit downloads (the weights of `zeit.ai`, embeddings, tiles) goes into one cache folder, by default `~/.cache/zeit`. On a cluster the home directory is small, and some compute nodes see it read-only: put the cache on the scratch space, in `~/.bashrc` so that jobs inherit it:

```bash
export ZEIT_CACHE_DIR=/scratch/$USER/zeit-cache
```

### With Apptainer

Each zeit version is published as a container image, which Apptainer (formerly Singularity) runs without root:

```bash
apptainer pull zeit.sif docker://ghcr.io/sacridini/zeit-cdts:2.3.0          # CPU
apptainer pull zeit-gpu.sif docker://ghcr.io/sacridini/zeit-cdts:2.3.0-gpu  # CUDA, for zeit.ai
apptainer exec zeit.sif zeit doctor
```

The image brings its own Python, GDAL and glibc, so it also runs where the wheels do not install (an older glibc). Apptainer mounts the home directory and the current one; add the others with `--bind /scratch`.

## First: `zeit doctor` in a job

What a job gets is not what the login node has. Ask a node:

```bash
srun --cpus-per-task=16 --time=5 zeit doctor
```

```text
zeit 2.3.0 (Python 3.12.8, Linux x86_64, glibc 2.28)
Threads (n_jobs=-1): 1 thread (OMP_NUM_THREADS=1; 16 of 128 processors available to this process; Slurm job 123456)
  OMP_NUM_THREADS=1, SLURM_CPUS_PER_TASK=16
  ! OMP_NUM_THREADS=1 leaves 15 of this job's processors idle (cluster modules often set it for MPI programs): set ZEIT_NUM_THREADS=16, which zeit reads first
Job: Slurm job 123456
Caches: /scratch/ana/zeit-cache (ZEIT_CACHE_DIR)
  weights     /scratch/ana/zeit-cache/weights      writable
  ...
Network:
  STAC, Earth Search                           NO: no answer in 5 s  (earth-search.aws.element84.com)
  ...
  ! This computer seems to have no internet access (...): `zeit fetch` downloads weights, embeddings and tiles.
```

Each line says what it found and where from. Here a module set `OMP_NUM_THREADS=1`, which zeit would obey, and the node has no internet access: the next sections deal with both.

## One node

Every algorithm already runs on all the processors a job is given: `n_jobs=-1` (the default, `--jobs` on the command line) counts the job's processors (`--cpus-per-task`, the cgroup Slurm confines the job to), not the node's. There is nothing to set:

```bash
#!/bin/bash
#SBATCH --job-name=landtrendr
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=4:00:00

source ~/venvs/zeit/bin/activate
export ZEIT_NUM_THREADS=$SLURM_CPUS_PER_TASK   # only if a module sets OMP_NUM_THREADS

zeit landtrendr nbr_stack.tif lt_out/ --max-segments 6
```

The same for Python code (`zeit.landtrendr(cube)`, `zeit.ccdc(...)`) and for a folder of tiles on one node (`zeit landtrendr tiles/ lt_out/ --tiles`): the tiles are processed one after another, each with all the job's processors.

## Many nodes: a job array over the tiles

A region in many tiles spreads over many nodes as a job array, one task per tile (or per few tiles). `zeit slurm` writes the script:

```bash
zeit slurm landtrendr tiles/ lt_out/ --cpus 16 --time 2:00:00 --partition normal --max-segments 8
```

```bash
#!/bin/bash
# Written by `zeit slurm`: landtrendr on 412 tiles of /scratch/ana/tiles, one per task of the array.
#   submit it:    sbatch /scratch/ana/lt_out/.zeit-tiles/landtrendr.sbatch
#   follow it:    zeit tiles status /scratch/ana/lt_out
# Each task takes its tiles from SLURM_ARRAY_TASK_ID and every processor it is given; finished tiles
# are kept, and a task stopped at its time limit requeues itself to continue.
# --mem: suggested for 10 blocks of 512 x 512 pixels at once (a row of the widest tile, 5000 columns of 40 bands); a smaller --chunk-size, or fewer --cpus, needs less.
#SBATCH --job-name=zeit-landtrendr
#SBATCH --array=0-411
#SBATCH --cpus-per-task=16
#SBATCH --mem=4G
#SBATCH --time=2:00:00
#SBATCH --partition=normal
#SBATCH --output=/scratch/ana/lt_out/.zeit-tiles/logs/slurm-%A_%a.out
#SBATCH --signal=B:TERM@120
#SBATCH --requeue

exec /home/ana/venvs/zeit/bin/python -m zeit.cli landtrendr /scratch/ana/tiles /scratch/ana/lt_out --tiles --requeue --max-segments 8
```

It is saved as `lt_out/.zeit-tiles/landtrendr.sbatch`; submit it with `sbatch`, or write and submit at once with `--submit`. What each line does:

- **The split.** Every task runs the same command; it finds its part from `SLURM_ARRAY_TASK_ID`, the tiles `k` with `k mod 412` equal to it. The first task keeps the split in `lt_out/.zeit-tiles/split.json`, so that a task submitted again later takes the same tiles.
- **The memory.** A task reads a tile in blocks of `--chunk-size` and writes it one row of blocks at a time, so it holds a row of blocks, at most one per processor: the width of the tiles and their bands count, not their height. `--mem` is suggested from that, measured for LandTrendr and CCDC: 4G above for 40 years, but 119G for CCDC on tiles 5000 pixels wide with 200 dates of 6 bands, where `--chunk-size 256` asks for 48G. Give your own with `--mem 24G`; `sacct -j <job> --format=JobID,MaxRSS` says what the first tasks used.
- **Fewer tasks.** `--tasks 50` makes 50 tasks of about 8 tiles each; `--max-running 20` lets at most 20 run at once (`--array=0-49%20`), for a fair share of the partition.
- **The environment.** The script runs zeit with the Python that wrote it, so the virtual environment needs no activation. `--setup "module load gdal"` adds a line before it, `--sbatch=--qos=long` another `#SBATCH` line.

### What is done, and what is left

```bash
zeit tiles status lt_out/
```

```text
landtrendr in lt_out: 412 tiles, 405 done, 3 failed, 4 not done
  failed: tile_031 (run 4242_31): MemoryError
          log: lt_out/.zeit-tiles/logs/tile_031.log
  ...
To process the 7 left: sbatch --array=31,40,88,101,117,203,377 lt_out/.zeit-tiles/landtrendr.sbatch
```

The last line submits again only the tasks that have tiles left; finished tiles are never processed twice. For a tile that failed for want of memory, raise `--mem` in the script (or write it again with `zeit slurm ... --mem 12G`) before resubmitting.

### Time limits

A task that reaches its time limit would be killed in the middle of a tile. With `--signal=B:TERM@120`, Slurm warns it two minutes before: zeit drops the tile at work (its unfinished files are deleted, the finished tiles stay) and, with `--requeue`, puts the task back in the queue, where it continues with its tiles left. A task that did not finish a single tile in its time does not requeue itself (its time limit is too short for one tile); `zeit tiles status` shows it.

### Other schedulers

PBS and LSF arrays do not tell a task how many there are: give the part with `--shard`, `zeit landtrendr tiles/ out/ --shard $PBS_ARRAY_INDEX/50` for an array of 50. Everything else (the split, the record, `zeit tiles status`, SIGTERM) is the same.

## Compute nodes without internet access

Download beforehand, on the login node, what zeit would download while running, into the cache the compute nodes see (`ZEIT_CACHE_DIR` on the scratch space):

```bash
zeit fetch weights prithvi-300m-tl                       # the weights of zeit.ai
zeit fetch tiles urls.txt                                # the tiles of a list of URLs
zeit fetch embeddings emb/ --source tessera --bbox -63.2 -10.1 -62.8 -9.7 --years 2020-2024
```

The job then runs `zeit landtrendr urls.txt out/ --tiles` (or the job array of `zeit slurm`) as if it had the network: each tile is in the cache. Tiles built from STAC or Earth Engine (`zeit.stac_tiles`, `zeit.gee_tiles`) are fetched the same way, in Python, with `zeit.fetch_tiles(sources)`. A node that cannot reach something zeit needs says so, with the `zeit fetch` command for it, instead of a connection error.

## GPUs for `zeit.ai`

```bash
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00

source ~/venvs/zeit/bin/activate        # installed with zeit-cdts[ai]
python train_tempcnn.py
```

With the container, the GPU image and `--nv`, which passes the node's driver into it:

```bash
apptainer exec --nv --bind /scratch zeit-gpu.sif python train_tempcnn.py
```

Fetch the weights of the pretrained models (Presto, Prithvi) on the login node first, with `zeit fetch weights`.

## Troubleshooting

### Zeit uses one thread

A module exported `OMP_NUM_THREADS=1` (common: it is the right value for MPI programs). Zeit warns once, and `zeit doctor` shows it. Set `ZEIT_NUM_THREADS`, which zeit reads first and other programs ignore: `export ZEIT_NUM_THREADS=$SLURM_CPUS_PER_TASK`.

### "Disk quota exceeded" in the home directory

The caches went to `~/.cache/zeit`. Set `ZEIT_CACHE_DIR` to the scratch space (see [Where the caches go](#where-the-caches-go)), and move the old cache there if it is worth keeping.

### `pip install` compiles zeit, and fails for want of Rust

There is no wheel for this computer: an older glibc than 2.28 (`ldd --version`: CentOS 7 has 2.17), or another architecture. Either use [Apptainer](#with-apptainer), or install Rust in your home directory, without root, and let pip compile zeit:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
source ~/.cargo/env
pip install zeit-cdts
```

Its dependencies (rasterio, PyTorch) need wheels for that glibc as well; on CentOS 7 the container is the simpler road.

### Earth Engine in a batch job

A job has no terminal to authenticate in: zeit says so instead of waiting for an answer. Authenticate once on the login node with `earthengine authenticate` (the credentials go to `~/.config/earthengine`, which the compute nodes see), or use a service account: `export GOOGLE_APPLICATION_CREDENTIALS=/home/ana/keys/zeit-sa.json`. A node without internet access cannot reach Earth Engine at all: build the tiles of `gee_tiles` beforehand with `zeit.fetch_tiles`.

### A task was killed for memory

`sacct -j <job>_<task> --format=JobID,State,MaxRSS` says `OUT_OF_MEMORY`. Raise `--mem`, or lower `--chunk-size` (each task holds one row of blocks of `--chunk-size` × `--chunk-size` pixels at a time, the whole history of each), and resubmit the tasks `zeit tiles status` lists.

## A cube larger than a node

A job array splits work that comes in tiles. One cube larger than a node's memory, computed by many nodes together, needs a Dask cluster across them (`dask-jobqueue`); zeit's functions take Dask-backed cubes (see [Parallel & Cloud Processing](parallel-cloud-processing.md)), but a guide for Dask on Slurm is still to come. Cutting the cube into tiles first (`gdal_retile.py`, or windows of `rioxarray`) and using a job array is usually simpler.
