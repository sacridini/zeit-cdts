#!/bin/bash
# Runs l2ps_dump (force-l2ps up to the end of radiometric_correction, with the intermediate results dumped) on a scene, as
# run_force.sh would run force-l2ps: the same parameter file (run_force.sh wrote it to OUT), 8 threads.
#   wsl -e bash -lc 'bash .../run_dump.sh SAFE_DIR OUT_DIR DUMP.bin'
set -eu
source ~/force-ref-work/env.sh
export OMP_NUM_THREADS=8 GDAL_NUM_THREADS=1
~/force-ref-work/real/l2ps_dump "$1" "$2/l2.prm" "$3" 2>&1 | grep -a "AOD:\|ok\|FAIL\|changed by"
