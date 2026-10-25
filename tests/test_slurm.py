"""Tiles on a cluster: the part of a job array task (and --shard), the split kept in the output
folder, the record of every run and `zeit tiles status`, SIGTERM before a time limit and the
requeue, `zeit slurm`'s script. Slurm itself is not needed: its variables are set by hand."""

import json
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest

import zeit
from zeit import cli
from zeit._tiles import _coordinator, _journal, _shard, _slurm, _tasks

from test_tiles import write_tile   # noqa: E402

SLURM = ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_ARRAY_TASK_MIN", "SLURM_ARRAY_TASK_MAX",
         "SLURM_ARRAY_TASK_STEP", "SLURM_ARRAY_TASK_COUNT", "PBS_JOBID", "PBS_ARRAY_INDEX", "PBS_ARRAYID",
         "LSB_JOBID", "LSB_JOBINDEX")


@pytest.fixture(autouse=True)
def no_job(monkeypatch, tmp_path):
    for name in SLURM:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ZEIT_CONFIG_DIR", str(tmp_path / "config"))


def array_task(monkeypatch, task, first=0, last=4, step=1, count=None, job="900"):
    monkeypatch.setenv("SLURM_JOB_ID", str(1000 + task))
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", job)
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", str(task))
    monkeypatch.setenv("SLURM_ARRAY_TASK_MIN", str(first))
    monkeypatch.setenv("SLURM_ARRAY_TASK_MAX", str(last))
    monkeypatch.setenv("SLURM_ARRAY_TASK_STEP", str(step))
    monkeypatch.setenv("SLURM_ARRAY_TASK_COUNT", str(count if count is not None else (last - first) // step + 1))


@pytest.fixture
def tiles(tmp_path):
    folder = tmp_path / "tiles"
    folder.mkdir()
    for i in range(5):
        write_tile(folder / f"tile_{i}.tif", i, 6, 7)
    return folder


# ---------------------------------------------------------------------------- the split

def test_shard_by_hand():
    assert _shard.parse("2/5") == _shard.Shard(2, 5, "--shard")
    for bad in ("5/5", "-1/3", "x", "1/0", "1"):
        with pytest.raises(ValueError):
            _shard.parse(bad)
    part = _shard.parse("1/3")
    assert [i for i in range(10) if part.takes(i)] == [1, 4, 7]


def test_a_slurm_array_task_finds_its_part_and_keeps_the_split(monkeypatch, tmp_path):
    array_task(monkeypatch, 3, first=1, last=9, step=2)        # --array=1-9:2, ids 1 3 5 7 9
    part = _shard.resolve(tmp_path)
    assert (part.index, part.count, part.source) == (1, 5, "Slurm job 900_3")
    assert part.task_id(4) == 9
    assert json.loads((tmp_path / _shard.STATE / _shard.SPLIT).read_text()) == {"count": 5, "first": 1, "step": 2}
    # submitted again for the tasks left (--array=7,9): the same split, from the folder
    array_task(monkeypatch, 9, first=7, last=9, step=2, count=2)
    assert _shard.resolve(tmp_path).index == 4
    array_task(monkeypatch, 4, first=4, last=4)
    with pytest.raises(ValueError, match="not one of the 5 tasks"):
        _shard.resolve(tmp_path)
    assert _shard.resolve(tmp_path, "0/2").count == 2       # --shard wins


def test_a_list_of_tasks_needs_a_split_from_before(monkeypatch, tmp_path):
    array_task(monkeypatch, 3, first=3, last=41, count=3)    # --array=3,17,41
    with pytest.raises(ValueError, match="not a range"):
        _shard.resolve(tmp_path)


def test_other_schedulers_are_asked_for_the_shard(monkeypatch, tmp_path):
    assert _shard.resolve(tmp_path) is None
    monkeypatch.setenv("PBS_ARRAY_INDEX", "3")
    with pytest.raises(ValueError, match=r"--shard \$PBS_ARRAY_INDEX/"):
        _shard.resolve(tmp_path)


def test_the_run_names(monkeypatch):
    assert "-" in _shard.run_name()                          # host-pid
    monkeypatch.setenv("SLURM_JOB_ID", "55")
    assert _shard.run_name() == "55"
    array_task(monkeypatch, 2)
    assert _shard.run_name() == "900_2"


def test_the_parts_cover_every_tile_once(tiles, tmp_path):
    seen = []
    for i in range(3):
        plan = _tasks.make_plan("mann-kendall", tiles, tmp_path / "out", {}, shard=_shard.parse(f"{i}/3"))
        seen += [t.stem for t in plan.tiles]
        assert plan.names == [f"tile_{k}" for k in range(5)]
    assert sorted(seen) == [f"tile_{k}" for k in range(5)]


# ---------------------------------------------------------------------------- runs and status

def test_array_tasks_share_an_output_folder_and_status_follows_them(tiles, tmp_path, monkeypatch):
    out = tmp_path / "out"
    (tiles / "tile_3.tif").write_bytes(b"not a raster")     # part 1 of 2: fails
    array_task(monkeypatch, 1, last=1)
    result = zeit.run_tiles("mann_kendall", tiles, out, local_cores=1, verbose=False)
    assert sorted(result.done) == ["tile_1"] and list(result.failed) == ["tile_3"]
    status = _journal.read_status(out)
    assert status.done == ["tile_1"] and list(status.failed) == ["tile_3"]
    assert status.missing == ["tile_0", "tile_2", "tile_4"] and not status.running
    assert Path(status.failed["tile_3"]["log"]).is_file() and status.failed["tile_3"]["run"] == "900_1"
    assert status.tasks(list(status.failed) + status.missing) == [0, 1]
    text = _journal.describe(status, script="job.sbatch")
    assert "sbatch --array=0-1 job.sbatch" in text and "failed: tile_3" in text

    array_task(monkeypatch, 0, last=1)
    zeit.run_tiles("mann_kendall", tiles, out, local_cores=1, verbose=False)
    status = _journal.read_status(out)
    assert status.done == ["tile_0", "tile_1", "tile_2", "tile_4"] and status.missing == []
    assert "sbatch --array=1 " in _journal.describe(status)
    assert not (out / _tasks.PARTIAL).exists()


def test_status_on_the_command_line(tiles, tmp_path, capsys):
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as exit_:
        cli.main(["tiles", "status", str(out)])
    assert exit_.value.code == 1
    zeit.run_tiles("mann_kendall", tiles, out, verbose=False, shard="0/5")
    with pytest.raises(SystemExit) as exit_:
        cli.main(["tiles", "status", str(out)])
    assert exit_.value.code == 3                              # some left
    assert "run the same command again" in capsys.readouterr().out
    for i in range(1, 5):
        zeit.run_tiles("mann_kendall", tiles, out, verbose=False, shard=f"{i}/5")
    with pytest.raises(SystemExit) as exit_:
        cli.main(["tiles", "status", str(out)])
    assert exit_.value.code == 0 and "Every tile is done" in capsys.readouterr().out


def test_ranges():
    assert _journal.ranges([9, 3, 4, 5, 17, 18]) == "3-5,9,17-18"
    assert _journal.ranges([]) == ""


# ---------------------------------------------------------------------------- SIGTERM and requeue

def slow_tiles(monkeypatch, started):
    """Tiles that take until they are cancelled."""
    def execute(command, tile_path, out_dir, args, shared, cores, cancel=None, log_path=None):
        started.set()
        cancel.wait(30)
        return False, 0.0, "cancelled"

    monkeypatch.setattr(_tasks, "execute", execute)


def test_sigterm_drops_the_tiles_at_work_and_keeps_the_record(tiles, tmp_path, monkeypatch):
    out = tmp_path / "out"
    started = threading.Event()
    slow_tiles(monkeypatch, started)
    threading.Thread(target=lambda: started.wait(30) and signal.raise_signal(signal.SIGTERM), daemon=True).start()
    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(_coordinator.Stopped) as stopped:
        zeit.run_tiles("mann_kendall", tiles, out, local_cores=1, verbose=False)
    assert stopped.value.signal_name == "SIGTERM" and stopped.value.done == []
    assert signal.getsignal(signal.SIGTERM) is previous      # the handler is put back
    assert not (out / _tasks.PARTIAL).exists()
    record = json.loads(next((out / _shard.STATE / _journal.RUNS).glob("*.json")).read_text())
    assert record["stopped"] == "SIGTERM" and record["done"] == []
    assert _journal.read_status(out).missing == [f"tile_{k}" for k in range(5)]


class FakeStopped:
    def __init__(self, done):
        self.done = done


def test_a_stopped_job_requeues_itself_when_it_made_progress(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(_slurm.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or
                        subprocess.CompletedProcess(cmd, 0, "", ""))
    assert cli._after_sigterm(FakeStopped(["a"]), requeue=False) == 143 and not calls
    array_task(monkeypatch, 7)
    assert cli._after_sigterm(FakeStopped([]), requeue=True) == 143 and not calls
    assert "no tile finished" in capsys.readouterr().out
    assert cli._after_sigterm(FakeStopped(["a"]), requeue=True) == 0
    assert calls == [["scontrol", "requeue", "900_7"]]


# ---------------------------------------------------------------------------- helpers and zeit slurm

def test_helpers_are_refused_inside_a_job(tiles, tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    with pytest.raises(RuntimeError, match="job array"):
        zeit.run_tiles("mann_kendall", tiles, tmp_path / "out", helpers=True, verbose=False)


def test_the_job_script(tiles, tmp_path, capsys):
    out = tmp_path / "out"
    cli.main(["slurm", "landtrendr", str(tiles), str(out), "--cpus", "8", "--mem", "12G", "--time", "2:00:00",
              "--partition", "cpu", "--max-running", "2", "--sbatch=--qos=long", "--setup", "module load gdal",
              "--max-segments", "4", "--jobs", "3"])
    path = out / _shard.STATE / "landtrendr.sbatch"
    expected = f"""#!/bin/bash
# Written by `zeit slurm`: landtrendr on 5 tiles of {tiles}, one per task of the array.
#   submit it:    sbatch {path}
#   follow it:    zeit tiles status {out}
# Each task takes its tiles from SLURM_ARRAY_TASK_ID and every processor it is given; finished tiles
# are kept, and a task stopped at its time limit requeues itself to continue.
#SBATCH --job-name=zeit-landtrendr
#SBATCH --array=0-4%2
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=2:00:00
#SBATCH --partition=cpu
#SBATCH --output={out / _shard.STATE / 'logs' / 'slurm-%A_%a.out'}
#SBATCH --signal=B:TERM@120
#SBATCH --requeue
#SBATCH --qos=long

module load gdal

exec {_q(sys.executable)} -m zeit.cli landtrendr {_q(tiles)} {_q(out)} --tiles --requeue --max-segments 4
"""
    assert path.read_text() == expected
    assert "Submit it with: sbatch" in capsys.readouterr().out
    assert _shard.read_split(out) == {"count": 5, "first": 0, "step": 1, "script": str(path)}
    assert (out / _shard.STATE / "logs").is_dir()          # Slurm does not make the log folder


def _q(path):
    import shlex

    return shlex.quote(str(path))


def test_the_job_script_splits_the_tiles_among_fewer_tasks(tiles, tmp_path):
    job = _slurm.write_script("mann-kendall", tiles, tmp_path / "out", {}, cpus=4, tasks=2)
    assert (job.tiles, job.tasks) == (5, 2)
    assert "#SBATCH --array=0-1\n" in job.text and "3 or fewer per task" in job.text
    assert "#SBATCH --mem=" in job.text and "#SBATCH --time" not in job.text


def test_the_suggested_memory_grows_with_the_processors():
    small, large = (int(_slurm.suggest_memory("ccdc", n)[:-1]) for n in (4, 32))
    assert 2 <= small < large
    assert int(_slurm.suggest_memory("ccdc", 32, chunk_size=256)[:-1]) < large


def test_the_suggested_memory_is_a_row_of_blocks_of_the_widest_tile():
    gb = lambda *args: int(_slurm.suggest_memory(*args)[:-1])          # noqa: E731
    # measured: 1.64 GB (LandTrendr, 2048 columns of 40 years), 19.2 GB (CCDC, 1024 columns of 200 x 6)
    assert 1.64 <= gb("landtrendr", 16, 512, 2048, 40) <= 1.64 * 2
    assert 19.2 <= gb("ccdc", 16, 512, 1024, 1200) <= 19.2 * 1.5
    assert gb("ccdc", 2, 512, 1024, 1200) == gb("ccdc", 64, 512, 1024, 1200)   # a row has two blocks
    assert gb("ccdc", 16, 256, 5000, 1200) < gb("ccdc", 16, 512, 5000, 1200)


def test_the_script_says_where_its_memory_comes_from(tiles, tmp_path):
    job = _slurm.write_script("landtrendr", tiles, tmp_path / "out", {}, cpus=16)
    assert "# --mem: suggested for 1 blocks of 512 x 512 pixels at once (a row of the widest tile" in job.text
    assert "# --mem" not in _slurm.write_script("landtrendr", tiles, tmp_path / "out", {}, cpus=16, mem="8G").text


def test_submitting_needs_sbatch(tmp_path, monkeypatch):
    monkeypatch.setattr(_slurm.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="login node"):
        _slurm.submit(tmp_path / "job.sbatch")


def test_bad_options_of_the_command_are_caught(tiles, tmp_path):
    with pytest.raises(SystemExit, match="zeit slurm"):
        cli.main(["slurm", "landtrendr", str(tiles), str(tmp_path / "out"), "--bogus", "1"])
    with pytest.raises(SystemExit, match="does not run per tile"):
        cli.main(["slurm", "embeddings", str(tiles), str(tmp_path / "out")])
