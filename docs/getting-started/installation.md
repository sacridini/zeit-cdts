# Installation

<p class="lead">One command installs Zeit with pre-compiled wheels for Windows, macOS and Linux. Build from source only if you want to change the C++ core or enable multi-threading on macOS.</p>

## Standard Installation

The easiest way to install Zeit is via Python's package manager, `pip`. We provide pre-compiled binaries (wheels) for Windows, macOS, and Linux, supporting Python 3.9 and newer. Because we distribute pre-compiled binaries, **you do not need a C++ compiler** installed on your machine for the standard installation.

```bash
pip install zeit-cdts
```

This command automatically installs all required Python dependencies, including `xarray`, `dask`, `scikit-learn`, `rasterio`, `torch`, and `pystac-client`.

!!! note "macOS: Apple Silicon vs. Intel"
    Prebuilt wheels are published for whichever architecture GitHub Actions' `macos-latest` runner uses at build time, which is Apple Silicon (`arm64`) as of this writing. If `pip install zeit-cdts` on an Intel Mac reports no matching distribution, pip will fall back to building from source automatically — see [Installing from Source](#installing-from-source) below for the compiler prerequisites that requires.

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

To read the [TESSERA embeddings](../tutorials/embeddings.md) with `zeit.load_embeddings(source="tessera")` (the `geotessera` library, which needs Python 3.12 or newer; the AlphaEarth embeddings need nothing extra):

```bash
pip install zeit-cdts[tessera]
```

For development and running tests, you can install the optional development dependencies:

```bash
pip install zeit-cdts[dev]
```

## Installing from Source

If you need to modify the C++ backend, use the latest unreleased features, or build the package on an unsupported architecture, you will need to install Zeit from the source.

### Requirements

*   Python 3.9+
*   A C++ Compiler supporting C++17 (GCC, Clang, or MSVC)
*   **macOS only:** the Xcode Command Line Tools provide the Clang compiler used to build the extension. Install them first if you haven't already:
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

This process will invoke `pybind11` and your system's C++ compiler to build the core statistical engines (`src/main.cpp`, `src/landtrendr.cpp`, `src/ccdc.cpp`) and link them with the Eigen3 library.

### Enabling OpenMP on macOS (Apple Silicon & Intel)

By default, the Apple Clang compiler does not include native support for OpenMP. As a result, when installing via standard Wheels or basic source installation, Zeit falls back to single-threaded mode for its C++ mathematical operations on macOS. (Note: Dask still parallelizes effectively at the chunk level).

If you want the maximum possible performance out of the C++ core on macOS, you can enable OpenMP by installing it via Homebrew and compiling Zeit from source:

1. Install the `libomp` library using Homebrew:
   ```bash
   brew install libomp
   ```

2. Point the compiler and linker at Homebrew's `libomp` — `setup.py`'s OpenMP detection compiles a test snippet against `<omp.h>`, which it won't find unless these are set, since Homebrew doesn't add `libomp` to the default include/lib search paths (it's keg-only):
   ```bash
   export CFLAGS="-I$(brew --prefix libomp)/include"
   export CXXFLAGS="-I$(brew --prefix libomp)/include"
   export LDFLAGS="-L$(brew --prefix libomp)/lib -lomp"
   ```

3. With those exported in the same shell, install Zeit from source. `setup.py` will now detect `libomp` and compile with OpenMP support:
   ```bash
   pip install --no-binary zeit-cdts zeit-cdts
   # or, if cloning from GitHub: pip install -e .
   ```

### OpenMP on Windows and Linux

Unlike macOS, Windows (MSVC) and Linux (GCC) ship with native OpenMP support, so no extra steps are needed — `setup.py` enables it automatically for both a standard `pip install zeit-cdts` (wheel) and a source install.

## GPU Acceleration for `zeit.ai`

The deep learning models in `zeit.ai` (`UTAE`, `LTAE`/`LightTAE`, `TempCNN`, `Siamese Change Detector`, `GeoFoundationViT`) are plain PyTorch `nn.Module`s and run on whatever device you move them to — none of them hard-code CUDA.

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
