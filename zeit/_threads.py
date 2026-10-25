"""The threads of ``n_jobs``, one rule for all of zeit (decided by the core, see
``rust/zeit-core/src/parallel.rs``).

``n_jobs`` > 0 is the number of threads. ``n_jobs`` <= 0 (or None) is, from the first that
applies: ``ZEIT_NUM_THREADS``; ``OMP_NUM_THREADS``; ``SLURM_CPUS_PER_TASK``, where the
processors this process may run on are more (a cluster that does not confine its jobs); the
processors this process may run on (its affinity and cgroup quota), all but one on a desktop
and all of them in a job of Slurm, PBS or LSF, or when the process is confined to fewer than
the machine has.
"""

import os
import warnings
from typing import NamedTuple, Optional

_warned = False


class Threads(NamedTuple):
    """What decides the threads of ``n_jobs=-1``."""
    threads: int                  # the threads of n_jobs <= 0
    all: int                      # with the processor kept free
    variable: Optional[str]       # the variable that gave the count, or None
    available: int                # processors of the process's affinity and cgroup quota
    machine: int                  # processors of the machine
    scheduler: Optional[str]      # Slurm, PBS, LSF, or None outside a job
    job: Optional[str]
    reserved: bool                # one processor is kept free (a desktop)

    def describe(self) -> str:
        """``"16 threads (16 of 128 processors available to this process; Slurm job 123456)"``."""
        where = []
        if self.variable is not None:
            where.append(f"{self.variable}={self.all}")
        if self.available < self.machine:
            where.append(f"{self.available} of {self.machine} processors available to this process")
        else:
            where.append(f"{self.machine} processors")
        if self.job is not None:
            where.append(f"{self.scheduler} job {self.job}")
        elif self.reserved:
            where.append("one kept free")
        return f"{self.threads} thread{'s' if self.threads != 1 else ''} ({'; '.join(where)})"

    def idle(self) -> int:
        """The processors of the job an OMP_NUM_THREADS lower than them leaves idle (0 outside a
        job, or when another variable gave the count): those the process may run on, or
        SLURM_CPUS_PER_TASK where it is fewer (a cluster that does not confine its jobs)."""
        if self.variable != "OMP_NUM_THREADS" or self.job is None:
            return 0
        return max(0, self.job_processors() - self.all)

    def job_processors(self) -> int:
        try:
            given = int(os.environ.get("SLURM_CPUS_PER_TASK", ""))
        except ValueError:
            given = 0
        return given if 0 < given < self.available else self.available


def detect() -> Threads:
    """The threads of ``n_jobs=-1`` now, and where they come from."""
    from . import _core

    return Threads(**_core.parallel.detect())


def resolve(n_jobs: Optional[int]) -> int:
    """The number of threads ``n_jobs`` asks for (see the module)."""
    if n_jobs is not None and int(n_jobs) > 0:
        return int(n_jobs)
    found = detect()
    _warn_low(found)
    return found.threads


def _warn_low(found: Threads) -> None:
    """Once: a job's processors left idle by an OMP_NUM_THREADS its environment set (cluster
    modules often export OMP_NUM_THREADS=1 for MPI programs)."""
    global _warned
    if _warned or not found.idle():
        return
    _warned = True
    job = found.job_processors()
    warnings.warn(f"zeit uses {found.all} thread{'s' if found.all != 1 else ''} because OMP_NUM_THREADS="
                  f"{found.all}, while this {found.scheduler} job may use {job} processors. To use "
                  f"them, set ZEIT_NUM_THREADS={job} (zeit reads it before OMP_NUM_THREADS) or pass "
                  f"n_jobs (--jobs on the command line).", RuntimeWarning, stacklevel=3)
