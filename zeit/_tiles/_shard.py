"""The part of the tiles that one task of a job array processes.

The tiles, in the order of the plan (sorted by name), go to the tasks in turn: tile ``i`` to
part ``i mod n``. A static split needs no lock in the file system (NFS and Lustre differ on
``O_EXCL``), and tiles are usually of similar size.

In a Slurm job array (``#SBATCH --array=0-49``) the part is the task's, from
``SLURM_ARRAY_TASK_ID``. The split (how many parts, the first task id, the step) is kept in
the output folder the first time (``.zeit-tiles/split.json``), so that the tasks submitted
again later, ``--array=3,17,41`` for those that failed, take the same tiles as before.
``--shard i/n`` gives the part by hand (``i`` from 0 to ``n - 1``), for other schedulers.
"""

import json
import os
import socket
from pathlib import Path
from typing import NamedTuple, Optional, Union

STATE = ".zeit-tiles"      # the run's records in the output folder: the split, the tasks' journals, logs
SPLIT = "split.json"


class ArrayTask(NamedTuple):
    """The task of a job array this process runs in."""
    scheduler: str
    job: str            # the array's job id
    task: int           # this task's id
    first: int          # the ids of the array: first, last, step, count
    last: int
    step: int
    count: int


class Shard(NamedTuple):
    """Part ``index`` of ``count`` (from 0)."""
    index: int
    count: int
    source: str         # "--shard" or the array's "Slurm job 123_4"
    first: int = 0      # the array task id of part 0, and between parts
    step: int = 1

    def task_id(self, part: int) -> int:
        """The array task id that runs ``part``."""
        return self.first + part * self.step

    def takes(self, position: int) -> bool:
        return position % self.count == self.index


def array_task() -> Optional[ArrayTask]:
    """The job array task of this process (Slurm), None outside one. PBS and LSF arrays give
    no count of their tasks: ValueError, asking for ``--shard``."""
    env = os.environ
    if env.get("SLURM_ARRAY_TASK_ID"):
        try:
            task = int(env["SLURM_ARRAY_TASK_ID"])
            first, last = int(env["SLURM_ARRAY_TASK_MIN"]), int(env["SLURM_ARRAY_TASK_MAX"])
            step = int(env.get("SLURM_ARRAY_TASK_STEP") or 1)
            count = int(env.get("SLURM_ARRAY_TASK_COUNT") or (last - first) // step + 1)
        except (KeyError, ValueError) as e:
            raise ValueError(f"the Slurm array variables are incomplete ({e}); give the part with --shard i/n") from None
        return ArrayTask("Slurm", env.get("SLURM_ARRAY_JOB_ID") or env.get("SLURM_JOB_ID", "?"), task, first, last,
                         max(step, 1), count)
    for name, variable in (("PBS", "PBS_ARRAY_INDEX"), ("PBS", "PBS_ARRAYID"), ("LSF", "LSB_JOBINDEX")):
        index = env.get(variable)
        if index and index != "0":
            raise ValueError(f"a {name} job array does not tell its tasks how many there are: give the part with "
                             f"--shard ${variable}/<number of tasks> (with indices from 0; from 1, "
                             f"--shard $(({variable} - 1))/<number of tasks>)")
    return None


def parse(text: str) -> Shard:
    """``"i/n"``: part ``i`` (from 0) of ``n``."""
    try:
        index, count = (int(v) for v in str(text).split("/"))
    except ValueError:
        raise ValueError(f"--shard takes i/n (part i, from 0, of n), got {text!r}") from None
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"--shard {text}: i goes from 0 to n - 1")
    return Shard(index, count, "--shard")


def read_split(output_dir: Union[str, Path]) -> Optional[dict]:
    try:
        split = json.loads((Path(output_dir) / STATE / SPLIT).read_text())
        return split if all(isinstance(split.get(k), int) for k in ("count", "first", "step")) else None
    except (OSError, ValueError, AttributeError):
        return None


def write_split(output_dir: Union[str, Path], count: int, first: int = 0, step: int = 1, **more) -> None:
    """Keep the split of the output folder, atomically (tasks starting together write the same)."""
    folder = Path(output_dir) / STATE
    folder.mkdir(parents=True, exist_ok=True)
    part = folder / f"{SPLIT}.{socket.gethostname()}-{os.getpid()}"
    part.write_text(json.dumps({"count": count, "first": first, "step": step, **more}, indent=1))
    os.replace(part, folder / SPLIT)


def resolve(output_dir: Union[str, Path], shard: Optional[str] = None) -> Optional[Shard]:
    """The part of the tiles this process runs: ``--shard`` when given, else this job array
    task's (with the split the output folder keeps), else None (all of them)."""
    if shard:
        return parse(shard)
    task = array_task()
    if task is None:
        return None
    split = read_split(output_dir)
    if split is None:
        if task.count != (task.last - task.first) // task.step + 1:
            raise ValueError(f"this Slurm array ({task.count} tasks between {task.first} and {task.last}) is not "
                             f"a range, and {output_dir} keeps no split of an earlier run: submit a range "
                             "(--array=0-49), or write the script with `zeit slurm`")
        split = {"count": task.count, "first": task.first, "step": task.step}
        write_split(output_dir, **split)
    offset = task.task - split["first"]
    if offset < 0 or offset % split["step"] or offset // split["step"] >= split["count"]:
        raise ValueError(f"array task {task.task} is not one of the {split['count']} tasks that split the tiles of "
                         f"{output_dir} (ids {split['first']} to {split['first'] + (split['count'] - 1) * split['step']}"
                         f"{', step ' + str(split['step']) if split['step'] != 1 else ''}, kept in {STATE}/{SPLIT} "
                         "since its first run; delete it to split them otherwise)")
    return Shard(offset // split["step"], split["count"], f"{task.scheduler} job {task.job}_{task.task}",
                 split["first"], split["step"])


def run_name() -> str:
    """This run's name among the runs writing into the same output folder: the job and array
    task (``123456_7``, or the job id), else the computer and process (``node12-4242``)."""
    env = os.environ
    job = env.get("SLURM_ARRAY_JOB_ID") or env.get("SLURM_JOB_ID") or env.get("PBS_JOBID") or env.get("LSB_JOBID")
    if job:
        task = env.get("SLURM_ARRAY_TASK_ID") or env.get("PBS_ARRAY_INDEX") or env.get("LSB_JOBINDEX")
        name = f"{job}_{task}" if task else job
    else:
        name = f"{socket.gethostname()}-{os.getpid()}"
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
