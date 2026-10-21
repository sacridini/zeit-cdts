# Docker

<p class="lead">Ready-made container images with Zeit, GDAL, PyTorch and JupyterLab. Use them to share an identical environment across a team, or to run Zeit workers on a cluster.</p>

## Official Docker Images

We provide two Dockerfiles, built the same way but on different base images:

* **`Dockerfile`** — based on the official PyTorch image with CUDA support (`pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel`), for GPU hosts.
* **`Dockerfile.cpu`** — based on the slim, multi-arch `python:3.11-slim` image with a CPU-only PyTorch wheel, for machines without an NVIDIA GPU. Because it has no CUDA dependency, this one builds and runs **natively on Apple Silicon (arm64) Macs**, with no emulation.

Both images include:
* A Rust toolchain to build Zeit's compiled core, and GDAL.
* Python dependencies and the Zeit package installed in editable mode.
* JupyterLab for interactive data science.

## Building and Running

To build the Rust core, install PyTorch, and launch a JupyterLab environment, you can use one of the two provided Compose files, depending on whether your host has an NVIDIA GPU.

### GPU hosts (NVIDIA + Linux/Windows)

`docker-compose.yml` reserves an NVIDIA GPU for the container (`deploy.resources.reservations.devices`), so it requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on the host in addition to Docker itself.

1. Ensure you have Docker, Docker Compose, and the NVIDIA Container Toolkit installed on your system.
2. Navigate to the root directory of the Zeit repository.
3. Run the following command:

```bash
docker-compose up --build
```

### CPU-only hosts (including macOS)

!!! warning
    The default `docker-compose.yml` will **fail to start** on any machine without an NVIDIA GPU — this includes every Mac, since Docker Desktop on macOS has no GPU passthrough. Use `docker-compose.cpu.yml` instead, which builds `Dockerfile.cpu` and drops the GPU reservation entirely.

```bash
docker-compose -f docker-compose.cpu.yml up --build
```

This builds a separate, CUDA-free image (`Dockerfile.cpu`) and runs it entirely on CPU. On Apple Silicon Macs it builds as a native `arm64` image — no `amd64` emulation, unlike running the default `Dockerfile` would require. `torch.cuda.is_available()` is `False` inside this container, since it has no GPU acceleration at all; it does not have access to the host's Metal/MPS GPU either, since Docker containers can't pass that through. For MPS acceleration on a Mac, use the native `pip install` path from the [installation guide](installation.md#gpu-acceleration-for-zeitai) instead of Docker.

!!! tip "Multi-threaded on every Mac chip"
    Zeit's compiled core starts its own threads, with no OpenMP runtime, so it runs on all the container's cores whatever the host chip, as it does in a native macOS install.

### Accessing the container

Either command builds the image and starts a Jupyter server. By default, the server runs on port 8888. You can access it by navigating to `http://localhost:8888` in your web browser.

The command used by the container is configured to allow root access and disables the automatic browser launch, making it ideal for headless server environments.
