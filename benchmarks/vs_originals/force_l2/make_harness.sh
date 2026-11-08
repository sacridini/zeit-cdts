#!/bin/bash
# Builds l2_harness: a main.c (Zeit's) compiled against FORCE 3.11.00's own headers and linked with the object files that
# FORCE's `make` built for `force-l2ps` (build/executables/force-l2ps.mk: aod-ll, atc-ll, atmo-ll, gas-ll, sunview-ll,
# topo-ll, radtran-ll, table-ll, brick_base-cl, quality-cl, imagefuns-cl, ... compiled by FORCE's Makefile with
# gcc -O3 -fopenmp, no -ffast-math, no -march), so that `radiometric_correction` runs as FORCE runs it. Not FORCE's, and
# the same on both sides: main.c replaces the sun position (sun-cl.o's `sunpos`: Zeit's is clean-room; a table of Zeit's
# values), the map to longitude and latitude (`warp_any_to_geo`), the warp of the DEM file (`warp_from_disc_to_known_brick`)
# and `brdf_factor` (brdf-ll.c is not ported); the objects that define those are linked after main.o with
# --allow-multiple-definition so that main.o's definitions win (brdf-ll.o is left out).
#
# GSL: the real GSL 2.7 of the conda env `force-ref`, linked STATICALLY with its own reference CBLAS (libgslcblas.a):
# conda-forge's libgslcblas.so is a symlink to OpenBLAS, whose dnrm2 rounds differently from GSL's. GDAL: the real libgdal
# 3.9.3 of the same env (GDALGridCreate of interpolate_aod_map and compile_l2_vzn; its AVX path for 'invdist').
#
#   wsl -e bash -lc 'bash /mnt/c/.../benchmarks/vs_originals/force_l2/make_harness.sh [FORCE_ROOT] [OUT_DIR]'
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${1:-$HOME/force-src}
OUT=${2:-$HOME/force-ref-work/l2}
source ~/force-ref-work/env.sh
GSLCBLAS_A=$CONDA_PREFIX/lib/libgslcblas.a
[ -f "$GSLCBLAS_A" ] && [ -f "$CONDA_PREFIX/lib/libgsl.a" ] || { echo "no static GSL in $CONDA_PREFIX/lib"; exit 1; }
[ -d "$ROOT/obj" ] || { echo "no $ROOT/obj: build FORCE first (make -j8 in $ROOT)"; exit 1; }
mkdir -p "$OUT"
MODS=$ROOT/src/modules
OBJ=$ROOT/obj
# the objects of force-l2ps, but for brdf-ll.o (replaced by main.c)
OBJS=""
for m in alloc-cl aod-ll atc-ll atmo-ll brick_base-cl brick_io-cl cite-cl cloud-ll coreg-ll coregfuns-ll cube-cl cube-ll \
  date-cl datesys-cl dir-cl enum-cl equi7-ll gas-ll gdalopt-cl glance7-ll imagefuns-cl konami-cl lock-cl meta-ll param-cl \
  param-ll quality-cl queue-cl radtran-ll read-cl read-ll resmerge-ll string-cl stats-cl sun-cl sunview-ll sys-cl table-ll \
  tile-cl topo-ll utils-cl vector-cl warp-cl; do
  OBJS="$OBJS $OBJ/$m.o"
done
gcc -O3 -Wall -fopenmp -I"$CONDA_PREFIX/include" -I"$MODS/lower-level" -c -o "$OUT/main.o" "$HERE/main.c"
g++ -fopenmp -o "$OUT/l2_harness" "$OUT/main.o" $OBJS \
  -Wl,--allow-multiple-definition \
  -L"$CONDA_PREFIX/lib" -l:libgsl.a "$GSLCBLAS_A" $(gdal-config --libs) -Wl,-rpath="$CONDA_PREFIX/lib" -lm
echo "built $OUT/l2_harness (GSL $(gsl-config --version))"
ldd "$OUT/l2_harness" | grep -i 'gsl\|blas' || echo "no shared GSL or BLAS in the executable (static)"
