"""zeit on a cluster: the threads of n_jobs=-1 (one rule, in Python and in the core), the
caches' folders, telling an offline computer from a server error, and `zeit doctor`."""

import os
import socket
import subprocess
import sys
import urllib.error
import warnings
from pathlib import Path

import pytest

from zeit import _cache, _core, _network, _threads
from zeit._tiles import _tasks

THREAD_VARIABLES = ("ZEIT_NUM_THREADS", "OMP_NUM_THREADS", "SLURM_CPUS_PER_TASK", "SLURM_JOB_ID", "PBS_JOBID",
                    "LSB_JOBID")


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in THREAD_VARIABLES + ("ZEIT_CACHE_DIR", "XDG_CACHE_HOME", "SLURM_ARRAY_TASK_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(_threads, "_warned", False)


# ---------------------------------------------------------------------------- threads

def test_a_positive_n_jobs_is_the_thread_count(monkeypatch):
    monkeypatch.setenv("ZEIT_NUM_THREADS", "7")
    assert _threads.resolve(3) == 3
    assert _threads.resolve(-1) == 7 and _threads.resolve(0) == 7 and _threads.resolve(None) == 7


def test_the_variables_in_their_order(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    assert _threads.resolve(-1) == 2
    assert _threads.detect().variable == "OMP_NUM_THREADS"
    monkeypatch.setenv("ZEIT_NUM_THREADS", "5")
    assert _threads.resolve(-1) == 5
    assert _threads.detect().variable == "ZEIT_NUM_THREADS"


def test_python_and_the_core_agree(monkeypatch):
    assert _core.parallel.threads(-1) == _threads.resolve(-1)
    monkeypatch.setenv("SLURM_JOB_ID", "123456")
    assert _core.parallel.threads(-1) == _threads.resolve(-1)
    monkeypatch.setenv("ZEIT_NUM_THREADS", "3")
    assert _core.parallel.threads(-1) == _threads.resolve(-1) == 3


def test_a_job_uses_every_processor(monkeypatch):
    desktop = _threads.detect()
    if not desktop.reserved:
        pytest.skip("this process is confined to some of the processors: nothing is kept free")
    assert desktop.threads == max(1, desktop.available - 1)
    monkeypatch.setenv("SLURM_JOB_ID", "123456")
    job = _threads.detect()
    assert job.threads == job.available and not job.reserved
    assert (job.scheduler, job.job) == ("Slurm", "123456")
    assert "Slurm job 123456" in job.describe()


def test_slurm_cpus_per_task_counts_where_nothing_confines_the_job(monkeypatch):
    available = _threads.detect().available
    if available < 2:
        pytest.skip("one processor")
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    assert _threads.resolve(-1) == 1
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", str(available + 4))   # more than it may use: the cgroup's
    assert _threads.resolve(-1) == available


def test_a_job_slowed_by_omp_num_threads_is_warned_once(monkeypatch):
    if _threads.detect().available < 2:
        pytest.skip("one processor")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _threads.resolve(-1) == 1           # outside a job: no warning
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    with pytest.warns(RuntimeWarning, match="ZEIT_NUM_THREADS"):
        assert _threads.resolve(-1) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _threads.resolve(-1)                       # once
    # a node that does not confine its jobs: the job's processors are SLURM_CPUS_PER_TASK
    monkeypatch.setattr(_threads, "_warned", False)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    found = _threads.detect()
    assert found.job_processors() == 2 and found.idle() == 1
    with pytest.warns(RuntimeWarning, match="ZEIT_NUM_THREADS=2 "):
        _threads.resolve(-1)
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    assert _threads.detect().idle() == 0


@pytest.mark.skipif(not hasattr(os, "sched_setaffinity"), reason="Linux only")
def test_an_affinity_of_two_processors_gives_two_threads():
    if len(os.sched_getaffinity(0)) < 3:
        pytest.skip("fewer than three processors")
    code = ("import os; os.sched_setaffinity(0, {0, 1}); "
            "from zeit import _threads, _core; print(_threads.resolve(-1), _core.parallel.threads(-1))")
    env = {k: v for k, v in os.environ.items() if k not in THREAD_VARIABLES}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    assert out.stdout.split() == ["2", "2"]        # confined: both of them, none kept free


def test_a_tile_process_is_capped_by_zeit_num_threads(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    env = _tasks.subprocess_env(6)
    assert env["ZEIT_NUM_THREADS"] == "6"
    assert env["OMP_NUM_THREADS"] == "1"           # the cluster's, left alone


# ---------------------------------------------------------------------------- caches

def test_one_root_for_every_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("ZEIT_CACHE_DIR", str(tmp_path / "scratch"))
    assert _cache.cache_root() == tmp_path / "scratch"
    assert _cache.caches() == {name: tmp_path / "scratch" / name for name in _cache.CACHES}
    monkeypatch.delenv("ZEIT_CACHE_DIR")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert _cache.cache_root() == tmp_path / "xdg" / "zeit"
    with pytest.raises(ValueError, match="no cache"):
        _cache.cache_dir("bogus")


def test_a_cache_an_older_zeit_filled_is_still_used(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    old = tmp_path / "home" / ".cache" / "zeit" / "weights"
    assert _cache.cache_dir("weights") == tmp_path / "xdg" / "zeit" / "weights"
    old.mkdir(parents=True)
    assert _cache.cache_dir("weights") == old
    monkeypatch.setenv("ZEIT_CACHE_DIR", str(tmp_path / "scratch"))     # an explicit root wins
    assert _cache.cache_dir("weights") == tmp_path / "scratch" / "weights"


def test_the_caches_go_where_the_root_says(monkeypatch, tmp_path):
    from zeit import _embeddings
    from zeit._tiles import _worker

    monkeypatch.setenv("ZEIT_CACHE_DIR", str(tmp_path))
    assert _embeddings._default_cache() == tmp_path / "embeddings"
    assert _worker.default_cache_dir() == tmp_path / "tiles"


def test_writable(tmp_path):
    assert _cache.writable(tmp_path / "not" / "yet")
    assert not _cache.writable(Path("/nonexistent-root-zeit") if os.name != "nt" else Path("Q:/zeit"))


# ---------------------------------------------------------------------------- network

def test_an_offline_computer_is_told_from_a_server_error():
    offline = urllib.error.URLError(socket.gaierror(-2, "Name or service not known"))
    assert _network.connection_failed(offline)
    assert _network.connection_failed(TimeoutError())
    try:
        try:
            raise ConnectionRefusedError(111, "refused")
        except OSError as e:
            raise RuntimeError("could not download") from e
    except RuntimeError as wrapped:
        assert _network.connection_failed(wrapped)
    server = urllib.error.HTTPError("https://x", 404, "Not Found", {}, None)
    assert not _network.connection_failed(server)
    assert not _network.connection_failed(ValueError("bad value"))
    text = _network.offline_message("huggingface.co", "zeit fetch weights presto")
    assert "login node" in text and "zeit fetch weights presto" in text


# ---------------------------------------------------------------------------- zeit doctor

def test_doctor_reports_threads_job_and_caches(monkeypatch, tmp_path):
    from zeit import _doctor

    monkeypatch.setenv("ZEIT_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    text = _doctor.report(network=False)
    assert "Threads (n_jobs=-1)" in text and "Slurm job 4242" in text
    assert f"Caches: {tmp_path} (ZEIT_CACHE_DIR)" in text
    for name in _cache.CACHES:
        assert str(tmp_path / name) in text
    assert "Network" not in text


def test_doctor_points_at_an_idle_job(monkeypatch):
    from zeit import _doctor

    if _threads.detect().available < 2:
        pytest.skip("one processor")
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    assert "set ZEIT_NUM_THREADS" in _doctor.report(network=False)


def test_doctor_on_the_command_line():
    out = subprocess.run([sys.executable, "-m", "zeit.cli", "doctor", "--no-network"], capture_output=True,
                         text=True, check=True)
    assert out.stdout.startswith("zeit ") and "Threads" in out.stdout


# ---------------------------------------------------------------------------- fetching beforehand

@pytest.fixture
def served(tmp_path):
    """Two tiles on a web server that can be switched off (the compute node's view of it)."""
    import functools
    import threading
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    import numpy as np
    import pandas as pd
    import xarray as xr
    from rasterio.transform import from_origin

    import zeit

    folder = tmp_path / "served"
    folder.mkdir()
    for seed in range(2):
        rng = np.random.default_rng(seed)
        values = (np.linspace(0.2, 0.8, 12)[:, None, None] + rng.normal(0, 0.05, (12, 8, 9))).astype("float32")
        tr = from_origin(500000.0 + 1000 * seed, 9000000.0, 30.0, 30.0)
        da = xr.DataArray(values, dims=("time", "y", "x"),
                          coords={"time": pd.date_range("2000-01-01", periods=12, freq="YS"),
                                  "y": tr.f - 30.0 * (np.arange(8) + 0.5),
                                  "x": tr.c + 30.0 * (np.arange(9) + 0.5)}).rio.write_crs("EPSG:32722")
        zeit.save_raster(da, folder / f"tile_{seed}.tif")

    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(folder)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    stopped = []

    def stop():
        if not stopped:
            server.shutdown()
            server.server_close()
            stopped.append(True)

    yield url, stop
    stop()


def test_tiles_fetched_beforehand_are_processed_offline(served, monkeypatch, tmp_path):
    import zeit

    monkeypatch.setenv("ZEIT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("ZEIT_CONFIG_DIR", str(tmp_path / "config"))
    url, stop = served
    urls = tmp_path / "tiles.txt"
    urls.write_text(f"{url}/tile_0.tif\n{url}/tile_1.tif\n")

    cached = zeit.fetch_tiles(urls, verbose=False)
    assert [p.parent for p in cached] == [tmp_path / "cache" / "tiles"] * 2
    assert sorted(p.name for p in (tmp_path / "cache" / "tiles").iterdir()) == sorted(p.name for p in cached)
    assert zeit.fetch_tiles(urls, verbose=False) == cached          # already there: nothing downloaded
    stop()                                                          # the compute node: no network
    result = zeit.run_tiles("mann_kendall", urls, tmp_path / "out", verbose=False)
    assert sorted(result.done) == ["tile_0", "tile_1"] and not result.failed


def test_a_tile_that_cannot_be_fetched_is_listed_and_the_others_kept(served, tmp_path):
    import zeit

    url, _ = served
    sources = zeit.tile_urls([f"{url}/tile_0.tif", f"{url}/missing.tif"])
    with pytest.raises(RuntimeError, match=r"1 of 2 tiles could not be fetched(.|\n)*\nmissing: "):
        zeit.fetch_tiles(sources, cache_dir=tmp_path / "cache", verbose=False)
    assert [p.suffix for p in (tmp_path / "cache").iterdir()] == [".tif"]     # tile_0 only, no leftovers
    with pytest.raises(ValueError, match="already here"):
        zeit.fetch_tiles(tmp_path, verbose=False)


def test_weights_are_named_as_zeit_fetch_weights_takes_them(monkeypatch, tmp_path):
    import hashlib

    from zeit import _weights

    names = list(_weights.published())
    assert names[0] == "presto" and "prithvi-300m-tl" in names and len(names) == 1 + len(_weights.PRITHVI_VARIANTS)
    with pytest.raises(ValueError, match="no weights bogus"):
        _weights.fetch_weights("presto", "bogus")

    # a copy made elsewhere, in the folder ZEIT_WEIGHTS names, is used as it is once its SHA-256 matches
    content = b"weights"
    monkeypatch.setattr(_weights, "PRESTO_SHA256", hashlib.sha256(content).hexdigest())
    (tmp_path / "default_model.pt").write_bytes(content)
    monkeypatch.setenv("ZEIT_WEIGHTS", str(tmp_path))
    assert _weights.fetch_weights("presto") == [tmp_path / "default_model.pt"]
    (tmp_path / "default_model.pt").write_bytes(b"something else")
    (tmp_path / "default_model.pt.sha256").unlink()
    with pytest.raises(RuntimeError, match="is not Presto's weights"):
        _weights.fetch_weights("presto")


def test_weights_out_of_reach_say_how_to_fetch_them(monkeypatch, tmp_path):
    from zeit import _weights

    def offline(*args, **kwargs):
        raise urllib.error.URLError(socket.gaierror(-3, "Temporary failure in name resolution"))

    monkeypatch.setenv("ZEIT_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("ZEIT_WEIGHTS", raising=False)
    monkeypatch.setattr(_weights.urllib.request, "urlopen", offline)
    with pytest.raises(RuntimeError, match="no internet access.*zeit fetch weights prithvi-300m-tl.*ZEIT_WEIGHTS"):
        _weights.fetch_weights("prithvi-300m-tl")
    assert not any(tmp_path.rglob("*.part"))


def test_earth_engine_in_a_batch_job_says_how_to_sign_in(monkeypatch, tmp_path):
    ee = pytest.importorskip("ee")
    from zeit.gee import auth

    asked = []
    monkeypatch.setattr(auth, "_can_ask", lambda: False)
    monkeypatch.setattr(ee, "Authenticate", lambda *a, **k: asked.append(True))
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    def no_credentials(*args, **kwargs):
        raise ee.EEException("Please authorize access to your Earth Engine account")

    monkeypatch.setattr(ee, "Initialize", no_credentials)
    with pytest.raises(RuntimeError, match="no terminal to authenticate in.*GOOGLE_APPLICATION_CREDENTIALS"):
        auth.initialize_gee("a-project")
    assert not asked                                                    # never waits for an answer

    def offline(*args, **kwargs):
        raise ee.EEException("Unable to find the server") from socket.gaierror(-3, "no DNS")

    monkeypatch.setattr(ee, "Initialize", offline)
    with pytest.raises(RuntimeError, match="could not reach Earth Engine.*zeit fetch tiles"):
        auth.initialize_gee("a-project")

    # a service account's key: its email and project
    key = tmp_path / "key.json"
    key.write_text('{"client_email": "zeit@a-project.iam.gserviceaccount.com", "project_id": "a-project"}')
    seen = {}
    monkeypatch.setattr(ee, "ServiceAccountCredentials", lambda email, path: ("credentials", email, path))
    monkeypatch.setattr(ee, "Initialize", lambda credentials=None, **kwargs: seen.update(kwargs, c=credentials))
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(key))
    auth.initialize_gee()
    assert seen["c"] == ("credentials", "zeit@a-project.iam.gserviceaccount.com", str(key))
    assert seen["project"] == "a-project" and seen["opt_url"] == auth.HIGH_VOLUME


def test_zeit_fetch_on_the_command_line(served, tmp_path):
    url, _ = served
    urls = tmp_path / "tiles.txt"
    urls.write_text(f"{url}/tile_0.tif\n")
    env = {**os.environ, "ZEIT_CACHE_DIR": str(tmp_path / "cache")}
    listed = subprocess.run([sys.executable, "-m", "zeit.cli", "fetch", "weights"], capture_output=True, text=True,
                            env=env)
    assert listed.returncode == 0 and "prithvi-600m-tl" in listed.stdout
    fetched = subprocess.run([sys.executable, "-m", "zeit.cli", "fetch", "tiles", str(urls)], capture_output=True,
                             text=True, env=env)
    assert fetched.returncode == 0, fetched.stdout + fetched.stderr
    assert "1 tiles in" in fetched.stdout and len(list((tmp_path / "cache" / "tiles").glob("*.tif"))) == 1
    bad = subprocess.run([sys.executable, "-m", "zeit.cli", "fetch", "weights", "bogus"], capture_output=True,
                         text=True, env=env)
    assert bad.returncode == 1 and "no weights bogus" in bad.stdout


def test_embeddings_and_stac_out_of_reach_say_how_to_fetch_them(monkeypatch):
    import zeit
    from zeit import _alphaearth, cube

    def offline(*args, **kwargs):
        raise urllib.error.URLError(socket.gaierror(-3, "Temporary failure in name resolution"))

    monkeypatch.setattr(_alphaearth, "parts", offline)
    with pytest.raises(RuntimeError, match="AlphaEarth embeddings.*zeit fetch embeddings --source alphaearth"):
        zeit.load_embeddings((-63.0, -10.0, -62.99, -9.99), source="alphaearth", years=2020)

    monkeypatch.setattr(cube.Client, "open", offline)
    with pytest.raises(RuntimeError, match="could not reach the STAC catalog.*zeit fetch tiles"):
        cube.build_time_series(bbox=[-63.0, -10.0, -62.99, -9.99], start_date="2020-01-01", end_date="2020-12-31")

    def refused(*args, **kwargs):
        raise urllib.error.HTTPError("https://example.org", 403, "Forbidden", {}, None)

    monkeypatch.setattr(_alphaearth, "parts", refused)        # an answer from the server: not offline
    with pytest.raises(urllib.error.HTTPError):
        zeit.load_embeddings((-63.0, -10.0, -62.99, -9.99), source="alphaearth", years=2020)
