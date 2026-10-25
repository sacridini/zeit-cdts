# Installation

<p class="lead">One command installs Zeit with pre-compiled wheels for Windows, macOS and Linux. Build from source only if you want to change the Rust core.</p>

## Standard Installation

The easiest way to install Zeit is via Python's package manager, `pip`. We provide pre-compiled binaries (wheels) for Windows, macOS, and Linux, supporting Python 3.9 and newer. Because we distribute pre-compiled binaries, **you do not need a compiler** (Rust or any other) installed on your machine for the standard installation. The wheels run the compiled core on all your cores on every system, macOS included.

```bash
pip install zeit-cdts
```

This command automatically installs all required Python dependencies, including `xarray`, `dask`, `scikit-learn`, `rasterio`, `torch`, and `pystac-client`.

!!! note "macOS: Apple Silicon vs. Intel"
    Prebuilt wheels are published for whichever architecture GitHub Actions' `macos-latest` runner uses at build time, which is Apple Silicon (`arm64`) as of this writing. If `pip install zeit-cdts` on an Intel Mac reports no matching distribution, pip will fall back to building from source automatically — see [Installing from Source](#installing-from-source) below for the Rust toolchain that requires.

## Verifying the Installation

Once installed, confirm the package and its CLI are working:

```bash
python -c "from importlib.metadata import version; print(version('zeit-cdts'))"
zeit --help
```

## Optional Dependencies

To look at maps and time series with [`zeit.plot`](../api/plot.md) (matplotlib for figures, anywidget for the interactive viewer in notebooks):

```bash
pip install zeit-cdts[plot]
pip install pywebview   # optional: show the viewer in a native window outside notebooks
```

The [embeddings](../tutorials/embeddings.md) of TESSERA and AlphaEarth need nothing extra: `geotessera`, which reads TESSERA, is installed with zeit on Python 3.12 or newer (it does not support older ones).

For development and running tests, you can install the optional development dependencies:

```bash
pip install zeit-cdts[dev]
```

## Caches

What zeit downloads is kept in one folder, with a subfolder per kind: `weights` (the pretrained weights of `zeit.ai`), `embeddings` (TESSERA and AlphaEarth downloads), `tiles` (the tiles a tile run built or received) and `map-tiles` (the basemaps of `zeit.plot`). The folder is, from the first that is set:

| | Folder |
| :--- | :--- |
| `ZEIT_CACHE_DIR` | as given |
| `XDG_CACHE_HOME` | `$XDG_CACHE_HOME/zeit` |
| Windows | `%LOCALAPPDATA%\zeit` |
| elsewhere | `~/.cache/zeit` |

On a cluster, point `ZEIT_CACHE_DIR` at the scratch space (`export ZEIT_CACHE_DIR=/scratch/$USER/zeit-cache`): the home directory's quota is small, and some compute nodes see it read-only. `zeit doctor` shows each folder and whether it can be written. A cache that zeit 2.2 or older filled in `~/.cache/zeit` keeps being used there until `ZEIT_CACHE_DIR` is set.

## Installing from Source

If you need to modify the Rust core, use the latest unreleased features, or build the package on an unsupported architecture, you will need to install Zeit from the source.

### Requirements

*   Python 3.9+
*   A Rust toolchain (stable), installed with [rustup](https://rustup.rs):
    ```bash
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh   # Linux and macOS
    ```
    On Windows, run `rustup-init.exe` from [rustup.rs](https://rustup.rs); it also offers to install the Visual Studio build tools Rust links with.
*   **macOS only:** the Xcode Command Line Tools, for the linker. Install them first if you haven't already:
    ```bash
    xcode-select --install
    ```

### Instructions

1. Clone the repository from GitHub:
   ```bash
   git clone https://github.com/sacridini/zeit-cdts.git
   cd zeit-cdts
   ```

2. Install the package in editable mode:
   ```bash
   pip install -e .
   ```

This builds the compiled core (`zeit._core`) from the Rust crates in `rust/` (`zeit-core`, the algorithms; `zeit-py`, their Python bindings) with `setuptools-rust`, in release mode. After changing the Rust code, rebuild it in place with:

```bash
python setup.py build_rust --inplace --release
```

The core is multi-threaded on every system, macOS included, with no extra library: it starts its own threads, so there is no OpenMP to install or enable.

## GPU Acceleration for `zeit.ai`

The deep learning models in `zeit.ai` (`UTAE`, `LTAE`/`LightTAE`, `TempCNN`, `Presto`, `Siamese Change Detector`, `Prithvi`) are plain PyTorch `nn.Module`s and run on whatever device you move them to — none of them hard-code CUDA.

*   **NVIDIA GPUs (Linux/Windows):** the standard `pip install zeit-cdts` installs a `torch` build with CUDA support where available. Use `torch.device("cuda")` as usual.
*   **Apple Silicon (any M-series chip):** PyTorch's Metal (`mps`) backend gives you native GPU acceleration on macOS — no CUDA or extra install needed, since it ships in the same `torch` package:
    ```python
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    ```
    Every tutorial in [AI & Deep Learning](../tutorials/ai.md) picks `"cuda"` vs. `"cpu"` in its example — swap in the snippet above on macOS to use the GPU.
*   **CPU fallback:** works everywhere, just slower.

!!! note "Docker and MPS"
    The [Docker image](docker.md) is built on a CUDA base image and has no access to Apple's Metal APIs. To use `mps` acceleration on macOS, install Zeit natively with `pip` rather than through Docker.
