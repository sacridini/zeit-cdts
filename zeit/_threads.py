"""The threads of ``n_jobs``, one rule for all of zeit (decided by the core, see
``rust/zeit-core/src/parallel.rs``).

``n_jobs`` > 0 is the number of threads. ``n_jobs`` <= 0 (or None) is, from the first that
applies: ``ZEIT_NUM_THREADS``; the threads of a dask.distributed worker (below); ``zeit config
threads``, outside a batch job; ``OMP_NUM_THREADS``; the job's processors from its
scheduler's variables (``SLURM_CPUS_PER_TASK``, ``SLURM_CPUS_ON_NODE``,
``SLURM_JOB_CPUS_PER_NODE``, ``NCPUS``, ``PBS_NUM_PPN``, ``LSB_DJOB_NUMPROC``), where the
processors this process may run on are more (a cluster that does not confine its jobs); the
processors this process may run on (its affinity and cgroup quota): all of them in a job of
Slurm, PBS or LSF, or when the process is confined to fewer than the machine has; an equal
share for each user on a machine several are logged in to at once (Remote Desktop, SSH);
all but one elsewhere. On a cluster's login node, outside a job, at most 4.

In a worker of dask.distributed, the worker's threads come before OMP_NUM_THREADS, which its
nanny sets to 1 for the BLAS libraries: the blocks that run there at once share the core's
pool of that many threads (one pool per count, per process), so they never ask for more.
"""
import os
import sys
import warnings
from typing import NamedTuple, Optional

_warned = False
_warned_shared = False
# What `variable` is when `zeit config threads` gave the count (the core's CONFIG).
CONFIG = "zeit config threads"


class Threads(NamedTuple):
    """What decides the threads of ``n_jobs=-1``."""
    threads: int                  # the threads of n_jobs <= 0
    all: int                      # with the processor kept free
    variable: Optional[str]       # the variable that gave the count, or None
    available: int                # processors of the process's affinity and cgroup quota
    machine: int                  # processors of the machine
    scheduler: Optional[str]      # Slurm, PBS, LSF, or None outside a job
    job: Optional[str]
    job_processors: int           # the job's (its scheduler's variables, or those available)
    reserved: bool                # one processor is kept free (a desktop)
    users: Optional[int]          # distinct users with an active session, None if unknown
    users_from: Optional[str]     # where they were read: utmp, logind, WTS
    shared: bool                  # the processors are divided among the users
    login_node: Optional[str]     # the scheduler of the cluster whose login node this is

    def describe(self) -> str:
        """``"16 threads (16 of 128 processors available to this process; Slurm job 123456)"``."""
        where = []
        if self.variable == CONFIG:
            where.append(f"{CONFIG} {self.all}")
        elif self.variable is not None:
            where.append(f"{self.variable}={self.all}")
        if self.available < self.machine:
            where.append(f"{self.available} of {self.machine} processors available to this process")
        else:
            where.append(f"{self.machine} processors")
        if self.job is not None:
            where.append(f"{self.scheduler} job {self.job}")
        elif self.shared:
            where.append(f"shared by {self.users} users with a session")
        elif self.reserved:
            where.append("one kept free")
        if self.login_node is not None:
            where.append(f"a {self.login_node} login node: at most 4")
        return f"{self.threads} thread{'s' if self.threads != 1 else ''} ({'; '.join(where)})"

    def idle(self) -> int:
        """The processors of the job an OMP_NUM_THREADS lower than them leaves idle (0 outside a
        job, or when another variable gave the count): those the process may run on, or fewer
        from the scheduler's variables (a cluster that does not confine its jobs)."""
        if self.variable != "OMP_NUM_THREADS" or self.job is None:
            return 0
        return max(0, self.job_processors - self.all)

    def lowered(self) -> Optional[str]:
        """Why n_jobs=-1 uses fewer threads than on a machine of one's own, if it does: a machine
        several users share, or a cluster's login node."""
        if self.login_node is not None:
            return (f"this looks like a login node of a {self.login_node} cluster (outside a job), "
                    f"which everyone shares: zeit uses {self.threads} of its {self.machine} processors. Run the "
                    f"analysis in a job (srun, sbatch, `zeit slurm`)")
        if self.shared:
            return (f"this machine is shared by {self.users} users with an active session ({self.users_from}): "
                    f"zeit uses {self.threads} of its {self.machine} processors, an equal share")
        return None

def detect() -> Threads:
    """The threads of ``n_jobs=-1`` now, and where they come from."""
    from . import _core

    return Threads(**_core.parallel.detect())


def resolve(n_jobs: Optional[int]) -> int:
    """The number of threads ``n_jobs`` asks for (see the module)."""
    if n_jobs is not None and int(n_jobs) > 0:
        return int(n_jobs)
    worker = _worker_threads()
    if worker and not os.environ.get("ZEIT_NUM_THREADS", "").strip():
        return worker
    found = detect()
    _warn_low(found)
    _warn_shared(found)
    return found.threads


def _worker_threads() -> Optional[int]:
    """The threads of the dask.distributed worker running this, or None outside one."""
    if "distributed" not in sys.modules:
        return None
    try:
        return int(sys.modules["distributed"].get_worker().state.nthreads) or None
    except Exception:  # noqa: BLE001 - not in a worker
        return None


def _warn_low(found: Threads) -> None:
    """Once: a job's processors left idle by an OMP_NUM_THREADS its environment set (cluster
    modules often export OMP_NUM_THREADS=1 for MPI programs)."""
    global _warned
    if _warned or not found.idle():
        return
    _warned = True
    job = found.job_processors
    warnings.warn(f"zeit uses {found.all} thread{'s' if found.all != 1 else ''} because OMP_NUM_THREADS="
                  f"{found.all}, while this {found.scheduler} job may use {job} processors. To use "
                  f"them, set ZEIT_NUM_THREADS={job} (zeit reads it before OMP_NUM_THREADS) or pass "
                  f"n_jobs (--jobs on the command line).", RuntimeWarning, stacklevel=3)


def _warn_shared(found: Threads) -> None:
    """Once: fewer threads on a machine others share, so that a slower analysis is not a
    mystery (its results are the same)."""
    global _warned_shared
    why = found.lowered()
    if _warned_shared or why is None:
        return
    _warned_shared = True
    warnings.warn(f"{why}. To change it, pass n_jobs (--jobs on the command line), set ZEIT_NUM_THREADS, or "
                  "run `zeit config threads N` once.", RuntimeWarning, stacklevel=3)
