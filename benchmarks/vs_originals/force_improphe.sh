#!/bin/bash
# The FORCE side of the ImproPhe comparison (force_improphe.py reads what this writes): generates the synthetic cube
# (force_improphe_cube.py), runs FORCE 3.11.00's CFIMP and L2IMP on it, one configuration per run, and copies the
# inputs and the outputs to the Windows folder the comparison reads (workdir("force_improphe")).
#
#   wsl -e bash -lc 'bash /mnt/c/.../benchmarks/vs_originals/force_improphe.sh ~/force-ref-work/imp-ref 6'
#
# Usage: force_improphe.sh <workdir> [nthread_compute=6]. NTHREAD_READ = NTHREAD_WRITE = 1 and OMP_NUM_THREADS = 8, so
# at most 8 threads in all. One run at a time, never next to a build.
#
# Each configuration is run twice, with FORCE's BLAS as built (A: OpenBLAS, OPENBLAS_NUM_THREADS=1, deterministic) and with
# GSL's reference CBLAS (B: LD_PRELOAD=$CONDA_PREFIX/lib/libgslcblas.so.0.0.0, whose cblas_* take precedence over OpenBLAS's),
# which is what zeit's port of gsl_eigen_symmv follows. Mind the file: conda-forge makes libgslcblas.so and libgslcblas.so.0
# symlinks to libcblas.so.3, that is OpenBLAS again (a "B" run with them is A: bit for bit the same output); the real GSL
# CBLAS is the file libgslcblas.so.0.0.0 (its dnrm2 is the scaled sum of squares of GSL's source, OpenBLAS's is not). Names: <cf|l2>_h<radius px>_<w4|w3>_<A|B>.
#   SIZE=297 (the tile's side, a multiple of 9), CF=3 (the field's pixels per side), CONFIGS="..." (a subset of names),
#   NOISE=0.1 (the cube's --band-noise: static noise per band and pixel, more principal components; use another workdir
#   and DEST for it).
set -u
WORK=$(mkdir -p "${1:?workdir}" && cd "$1" && pwd)
NCOMP=${2:-6}
HERE=$(cd "$(dirname "$0")" && pwd)
DEST=${DEST:-/mnt/c/Users/eduar/AppData/Local/Temp/zeit-vs-force_improphe}
SIZE=${SIZE:-297}
CF=${CF:-3}
CONFIGS=${CONFIGS:-}
NOISE=${NOISE:-0}
source ~/force-ref-work/env.sh
export OMP_NUM_THREADS=8 GDAL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
GSLCBLAS=$CONDA_PREFIX/lib/libgslcblas.so.0.0.0
# FORCE 3.11.00's L2IMP does not run as released: param-hl.c never fills the target sensor, main product and quality product
# of the secondary (SENSORS_LOWRES) sensors, so it stops at "Item with key (null) not found in JSON" and, past that, the
# reading of the Landsat scenes segfaults. HL_L2 is a copy of FORCE's build tree where param-hl.c fills them (the same
# flags, the same objects, nothing of the improphe code touched): see force_improphe_l2fix.sh. CFIMP runs with the
# installed binary.
HL_L2=${HL_L2:-$HOME/force-ref-work/force-l2fix/bin/force-higher-level}
[ -f "$GSLCBLAS" ] && [ ! -L "$GSLCBLAS" ] || { echo "no real $GSLCBLAS"; exit 1; }

mkdir -p "$WORK/prm" "$WORK/out" "$WORK/logs"
if [ ! -f "$WORK/ard/X0000_Y0000/.done" ]; then
    rm -rf "$WORK/ard" "$WORK/coarse"
    python -I "$HERE/force_improphe_cube.py" "$WORK" --size "$SIZE" --cf-factor "$CF" --seed 1 --band-noise "$NOISE" || exit 1
    touch "$WORK/ard/X0000_Y0000/.done"
fi
TILE=$(awk '/^TILE_SIZE_X/ {printf "%d", $3}' "$WORK/ard/datacube-definition.prj")
[ -f "$WORK/cf-template.prm" ] || force-parameter "$WORK/cf-template.prm" CFIMP >/dev/null
[ -f "$WORK/l2-template.prm" ] || force-parameter "$WORK/l2-template.prm" L2IMP >/dev/null

cat > "$WORK/mkprm.py" <<'EOF'
import re, sys
tpl, out, *kv = sys.argv[1:]
text = open(tpl).read()
for item in kv:
    k, v = item.split("=", 1)
    text, n = re.subn(r"^%s\s*=.*$" % re.escape(k), lambda m: "%s = %s" % (k, v), text, flags=re.M)
    if n != 1:
        sys.exit("key %s found %d times in %s" % (k, n, tpl))
open(out, "w").write(text)
EOF

COMMON=(DIR_LOWER="$WORK/ard" NTHREAD_READ=1 NTHREAD_WRITE=1 NTHREAD_COMPUTE="$NCOMP" PRETTY_PROGRESS=FALSE
        RESOLUTION=10 CHUNK_SIZE="$TILE $TILE" SENSORS=SEN2A TARGET_SENSOR=SEN2A DATE_RANGE="2018-01-01 2019-12-31"
        KERNEL_TEXT=30)
W4="1 91 181 271 365"
W3="1 121 241 365"

TIMES=$WORK/times.tsv
: > "$TIMES"
printf 'config\tseconds\texit\n' >> "$TIMES"
FAILED=0

# run <cf|l2> <radius px> <w4|w3> <A|B>
run() {
    local mod=$1 h=$2 win=$3 blas=$4
    local name=${mod}_h${h}_${win}_${blas}
    if [ -n "$CONFIGS" ]; then
        case " $CONFIGS " in *" $name "*) ;; *) return ;; esac
    fi
    local o=$WORK/out/$name wv=$W4
    [ "$win" = w3 ] && wv=$W3
    rm -rf "$o"; mkdir -p "$o"
    local extra
    if [ "$mod" = cf ]; then
        extra=(DIR_COARSE="$WORK/coarse" BASE_COARSE=FIELD.tif COARSE_1ST_YEAR=2018 COARSE_PREDICT_YEARS="2018 2019")
        tpl=$WORK/cf-template.prm
    else
        extra=(SENSORS_LOWRES=LND08)
        tpl=$WORK/l2-template.prm
    fi
    python -I "$WORK/mkprm.py" "$tpl" "$WORK/prm/$name.prm" "${COMMON[@]}" DIR_HIGHER="$o" DIR_PROVENANCE="$o" \
        KERNEL_SIZE=$((h * 10)) SEASONAL_WINDOW="$wv" "${extra[@]}" || { echo "mkprm failed for $name"; FAILED=1; return; }
    local t0 t1 rc hl=force-higher-level
    [ "$mod" = l2 ] && hl=$HL_L2
    t0=$(date +%s.%N)
    if [ "$blas" = B ]; then
        LD_PRELOAD="$GSLCBLAS" "$hl" "$WORK/prm/$name.prm" > "$WORK/logs/$name.log" 2>&1
    else
        "$hl" "$WORK/prm/$name.prm" > "$WORK/logs/$name.log" 2>&1
    fi
    rc=$?
    t1=$(date +%s.%N)
    printf '%s\t%.3f\t%d\n' "$name" "$(echo "$t1 - $t0" | bc -l)" "$rc" >> "$TIMES"
    if [ $rc -ne 0 ] || grep -q "no input or output detected" "$WORK/logs/$name.log" || ! ls "$o"/X0000_Y0000/*_IMP.tif >/dev/null 2>&1; then
        echo "run $name failed ($rc), see $WORK/logs/$name.log"; FAILED=1; return
    fi
    echo "done $name ($rc)"
}

for mod in cf l2; do
    for h in 10 20; do
        for win in w4 w3; do
            for blas in A B; do
                run $mod $h $win $blas
            done
        done
    done
done

# Copy the inputs (once) and the outputs.
mkdir -p "$DEST"
if [ ! -f "$DEST/.data-$SIZE-$CF" ]; then
    rm -rf "$DEST/ard" "$DEST/coarse" "$DEST"/.data-*
    cp -r "$WORK/ard" "$WORK/coarse" "$WORK/truth.npz" "$DEST/"
    touch "$DEST/.data-$SIZE-$CF"
fi
mkdir -p "$DEST/out" "$DEST/prm"
for d in "$WORK"/out/*/; do
    name=$(basename "$d")
    rm -rf "$DEST/out/$name"
    cp -r "$d" "$DEST/out/$name"
    cp "$WORK/prm/$name.prm" "$DEST/prm/"
done
cp "$WORK/times.tsv" "$DEST/"
echo "copied to $DEST"
cat "$TIMES"
exit $FAILED
