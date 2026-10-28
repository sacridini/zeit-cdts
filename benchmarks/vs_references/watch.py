"""The quiet-machine watchdog of the vs_references benchmarks (started by common.py, not by hand).

Every half second it appends a line ``time cargo other_load`` to a log: ``cargo`` is 1 when a cargo or rustc
process exists, ``other_load`` the processor load in percent of the whole machine left once the benchmark's own
process is taken out. It runs as a process of its own, on the last processor, so that its polling never takes the
GIL or the processor of the timed code, and it exits as soon as the benchmark process does.

    python watch.py <benchmark pid> <log file>
"""
import os
import sys
import time

import psutil

BUILDERS = {"cargo.exe", "rustc.exe", "cargo", "rustc"}


def main(pid, log):
    me = psutil.Process()
    try:
        me.cpu_affinity([psutil.cpu_count() - 1])
        me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:
        pass
    try:
        bench = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    ncpu = psutil.cpu_count()
    psutil.cpu_percent(None)
    bench.cpu_percent(None)
    with open(log, "a", buffering=1) as f:
        while True:
            time.sleep(0.5)
            if not bench.is_running():
                return
            try:
                own = bench.cpu_percent(None) / ncpu
            except psutil.NoSuchProcess:
                return
            total = psutil.cpu_percent(None)
            cargo = 0
            for p in psutil.process_iter(["name"]):
                if (p.info["name"] or "").lower() in BUILDERS:
                    cargo = 1
                    break
            f.write(f"{time.time():.3f} {cargo} {max(total - own, 0.0):.1f}\n")


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2])
