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
