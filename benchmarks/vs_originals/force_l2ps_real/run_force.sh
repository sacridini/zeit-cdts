#!/bin/bash
# Runs FORCE 3.11.00's force-l2ps (the conda env `force-ref`, built in WSL) on a Sentinel-2 L1C .SAFE: no reprojection or
# datacube (DO_REPROJ = DO_TILE = FALSE: the products stay on the grid of the scene), RES_MERGE = NONE (or the fifth argument:
# REGRESSION, IMPROPHE), the atmospheric, topographic and (optionally) BRDF corrections with the DEM, 8 threads in total (NPROC 1,
# NTHREAD 8).
#
#   wsl -e bash -lc 'bash /mnt/c/.../force_l2ps_real/run_force.sh SAFE_DIR DEM_FILE OUT_DIR BRDF(TRUE|FALSE) [RES_MERGE [NTHREAD]]'
#
# Prints the stdout of force-l2ps (it holds the AOD and the number of targets) and the wall time.
set -eu
SAFE=$1
DEM=$2
OUT=$3
BRDF=${4:-FALSE}
RESMERGE=${5:-NONE}
NTHREAD=${6:-8}
HERE=$(cd "$(dirname "$0")" && pwd)
source ~/force-ref-work/env.sh
export OMP_NUM_THREADS=$NTHREAD GDAL_NUM_THREADS=1 CARGO_BUILD_JOBS=8 RUST_TEST_THREADS=8 RAYON_NUM_THREADS=8
rm -rf "$OUT"
mkdir -p "$OUT/log" "$OUT/prov" "$OUT/tmp"
sed -e 's/\r$//' -e "s#@OUT@#$OUT#g" -e "s#@DEM@#$DEM#g" -e "s#@BRDF@#$BRDF#g" -e "s#@RESMERGE@#$RESMERGE#g" -e "s#@NTHREAD@#$NTHREAD#g" "$HERE/l2.prm.in" > "$OUT/l2.prm"
start=$(date +%s.%N)
force-l2ps "$SAFE" "$OUT/l2.prm" 2>&1 | tee "$OUT/force-l2ps.log"
end=$(date +%s.%N)
echo "wall seconds: $(echo "$end - $start" | bc)" | tee -a "$OUT/force-l2ps.log"
ls -la "$OUT"/*/
