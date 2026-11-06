#!/bin/bash
# The FORCE side of the ImproPhe timing (force_improphe_timing.py reads what this writes): generates the measurement cube
# (force_improphe_cube.py, 153 x 153 pixels of 10 m, 2 Landsat 8 dates a year), runs FORCE 3.11.00's CFIMP and L2IMP on it,
# one configuration per run, several times, and keeps the time of FORCE's own compute stage of every run.
#
#   wsl -e bash -lc 'bash /mnt/c/.../benchmarks/vs_originals/force_improphe_timing.sh ~/force-ref-work/imp-bench 6 [config...]'
#
# Usage: force_improphe_timing.sh <workdir> <nthread_compute> [config...]   (a config is cf_h<radius px> or l2_h<radius px>)
#   REPS="1 2 3", SIZE=153 (the tile's side, a multiple of 9), CF=3. One config after the other, never in parallel.
# NTHREAD_READ = NTHREAD_WRITE = 1, so the threads in all are nthread_compute + 2 (<= 8 with 6); OPENBLAS_NUM_THREADS = 1.
#
# The compute stage's time, as in force_tsa_timing.sh: force_time_ms.c makes FORCE's time() count milliseconds, so the
# "Virtual C-time" of its end summary is read in milliseconds. For CFIMP/L2IMP the stage (H/tasks-hl.c compute_higher_level)
# is screen_qai of the inputs, the seasonal means, the PCA, the standardization, the textures and the kernel, with no reading and
# no writing. OMP_NUM_THREADS = nthread_compute + 2 (at least 4), see force_tsa_timing.sh.
# L2IMP runs with the patched build of force_improphe_l2fix.sh (see force_improphe.sh).
#
# Output: $WORK/compute.tsv (config, nthread_compute, rep, wall_s, input_ms, compute_ms, output_ms) and, in $DEST, the cube
# (ard, coarse) and compute.tsv.
set -u
WORK=$(mkdir -p "${1:?workdir}" && cd "$1" && pwd)
NCOMP=${2:?nthread_compute}
shift 2
CONFIGS=("$@")
HERE=$(cd "$(dirname "$0")" && pwd)
DEST=${DEST:-/mnt/c/Users/eduar/AppData/Local/Temp/zeit-vs-force_improphe/bench}
SIZE=${SIZE:-153}
CF=${CF:-3}
REPS=${REPS:-"1 2 3"}
source ~/force-ref-work/env.sh
OMP=$((NCOMP + 2)); [ $OMP -ge 4 ] || OMP=4
export OMP_NUM_THREADS=$OMP GDAL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
HL_L2=${HL_L2:-$HOME/force-ref-work/force-l2fix/bin/force-higher-level}

mkdir -p "$WORK/prm" "$WORK/out" "$WORK/logs"
if [ ! -f "$WORK/ard/X0000_Y0000/.done" ]; then
    rm -rf "$WORK/ard" "$WORK/coarse"
    python -I "$HERE/force_improphe_cube.py" "$WORK" --size "$SIZE" --cf-factor "$CF" --seed 1 --l8-per-year 2 || exit 1
    touch "$WORK/ard/X0000_Y0000/.done"
fi
SHIM="$WORK/force_time_ms.so"
[ -f "$SHIM" ] && [ "$SHIM" -nt "$HERE/force_time_ms.c" ] || gcc -O2 -shared -fPIC -o "$SHIM" "$HERE/force_time_ms.c" || exit 1
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
cat > "$WORK/decode.py" <<'EOF'
import re, sys
text = open(sys.argv[1]).read()
out = []
for label in ("Virtual I-time", "Virtual C-time", "Virtual O-time"):
    m = re.search(label + r":\s+(\d+)y (\d+)m (\d+)d (\d+)h (\d+)m (-?\d+)s", text)
    if not m:
        sys.exit("no %s in %s" % (label, sys.argv[1]))
    y, mo, d, h, mi, s = map(int, m.groups())
    out.append(y * 946080000 + mo * 2592000 + d * 86400 + h * 3600 + mi * 60 + s)
print(*out)
EOF

COMMON=(DIR_LOWER="$WORK/ard" NTHREAD_READ=1 NTHREAD_WRITE=1 NTHREAD_COMPUTE="$NCOMP" PRETTY_PROGRESS=FALSE
        RESOLUTION=10 CHUNK_SIZE="$TILE $TILE" SENSORS=SEN2A TARGET_SENSOR=SEN2A DATE_RANGE="2018-01-01 2019-12-31"
        KERNEL_TEXT=30 SEASONAL_WINDOW="1 91 181 271 365")

[ -f "$WORK/compute.tsv" ] || printf 'config\tncomp\trep\twall_s\tinput_ms\tcompute_ms\toutput_ms\n' > "$WORK/compute.tsv"
FAILED=0

# run <cf_hN|l2_hN> <rep>
run() {
    local name=$1 rep=$2
    local mod=${name%%_*} h=${name##*_h}
    local o=$WORK/out/$name log=$WORK/logs/${name}_t${NCOMP}_r${rep}.log tpl extra hl=force-higher-level
    rm -rf "$o"; mkdir -p "$o"
    if [ "$mod" = cf ]; then
        tpl=$WORK/cf-template.prm
        extra=(DIR_COARSE="$WORK/coarse" BASE_COARSE=FIELD.tif COARSE_1ST_YEAR=2018 COARSE_PREDICT_YEARS="2018 2019")
    else
        tpl=$WORK/l2-template.prm; hl=$HL_L2
        extra=(SENSORS_LOWRES=LND08)
    fi
    python -I "$WORK/mkprm.py" "$tpl" "$WORK/prm/$name.prm" "${COMMON[@]}" DIR_HIGHER="$o" DIR_PROVENANCE="$o" \
        KERNEL_SIZE=$((h * 10)) "${extra[@]}" || { echo "mkprm failed for $name"; FAILED=1; return; }
    local t0 t1 rc
    t0=$(date +%s.%N)
    LD_PRELOAD="$SHIM" "$hl" "$WORK/prm/$name.prm" > "$log" 2>&1
    rc=$?
    t1=$(date +%s.%N)
    if [ $rc -ne 0 ] || grep -q "no input or output detected" "$log" || ! ls "$o"/X0000_Y0000/*_IMP.tif >/dev/null 2>&1; then
        echo "run $name failed ($rc), see $log"; FAILED=1; return
    fi
    local ms
    ms=$(python -I "$WORK/decode.py" "$log") || { FAILED=1; return; }
    printf '%s\t%d\t%d\t%.3f\t%s\n' "$name" "$NCOMP" "$rep" "$(echo "$t1 - $t0" | bc -l)" "${ms// /$'\t'}" >> "$WORK/compute.tsv"
    echo "done $name t=$NCOMP rep=$rep: wall $(echo "$t1 - $t0" | bc -l | cut -c1-6) s, I/C/O ms: $ms"
}

[ ${#CONFIGS[@]} -gt 0 ] || CONFIGS=(cf_h10 cf_h20 cf_h50 l2_h10 l2_h20 l2_h50)
for name in "${CONFIGS[@]}"; do
    for rep in $REPS; do
        run "$name" "$rep"
    done
done

mkdir -p "$DEST"
if [ ! -f "$DEST/.data-$SIZE-$CF" ]; then
    rm -rf "$DEST/ard" "$DEST/coarse" "$DEST"/.data-*
    cp -r "$WORK/ard" "$WORK/coarse" "$WORK/truth.npz" "$DEST/"
    touch "$DEST/.data-$SIZE-$CF"
fi
cp "$WORK/compute.tsv" "$DEST/"
echo "copied to $DEST"
exit $FAILED
