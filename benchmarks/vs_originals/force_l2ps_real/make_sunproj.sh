#!/bin/bash
# Builds sunproj (sunproj.c) against FORCE 3.11.00's headers and the objects FORCE's make built for force-l2ps, as
# force_l1/make_harness.sh does (brdf-ll.o is left out: sunproj.c has a stub of brdf_factor, which nothing calls).
#
#   wsl -e bash -lc 'bash /mnt/c/.../force_l2ps_real/make_sunproj.sh [FORCE_ROOT] [OUT_DIR]'
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${1:-$HOME/force-src}
OUT=${2:-$HOME/force-ref-work/real}
source ~/force-ref-work/env.sh
export CARGO_BUILD_JOBS=8 RAYON_NUM_THREADS=8
GSLCBLAS_A=$CONDA_PREFIX/lib/libgslcblas.a
[ -d "$ROOT/obj" ] || { echo "no $ROOT/obj: build FORCE first"; exit 1; }
mkdir -p "$OUT"
MODS=$ROOT/src/modules
OBJ=$ROOT/obj
OBJS=""
for m in alloc-cl aod-ll atc-ll atmo-ll brick_base-cl brick_io-cl cite-cl cloud-ll coreg-ll coregfuns-ll cube-cl cube-ll \
  date-cl datesys-cl dir-cl enum-cl equi7-ll gas-ll gdalopt-cl glance7-ll imagefuns-cl konami-cl lock-cl meta-ll param-cl \
  param-ll quality-cl queue-cl radtran-ll read-cl read-ll resmerge-ll string-cl stats-cl sun-cl sunview-ll sys-cl table-ll \
  tile-cl topo-ll utils-cl vector-cl warp-cl; do
  OBJS="$OBJS $OBJ/$m.o"
done
gcc -O2 -Wall -fopenmp -I"$CONDA_PREFIX/include" -I"$MODS/lower-level" -c -o "$OUT/sunproj.o" "$HERE/sunproj.c"
g++ -fopenmp -o "$OUT/sunproj" "$OUT/sunproj.o" $OBJS -Wl,--allow-multiple-definition \
  -L"$CONDA_PREFIX/lib" -l:libgsl.a "$GSLCBLAS_A" $(gdal-config --libs) -Wl,-rpath="$CONDA_PREFIX/lib" -lm
echo "built $OUT/sunproj"
