#!/bin/bash
# The FORCE side of the timing comparison (force_tsa_timing.py reads what this writes): generates the measurement
# cube (force_tsa_cube.py, 500 x 500 pixels by default), runs FORCE 3.11.00's higher-level TSA and CSO on it, one
# configuration per run, several times, and keeps the time of FORCE's own compute stage of every run.
#
#   wsl -e bash -lc 'bash /mnt/c/.../benchmarks/vs_originals/force_tsa_timing.sh ~/force-ref-work/tsa-bench 6 [config...]'
#
# Usage: force_tsa_timing.sh <workdir> <nthread_compute> [config...]   (no config: all of them)
#   REPS="1 2 3" (env) the repetitions to run, SIZE=500 the cube's side; one config after the other, never in parallel.
# NTHREAD_READ = NTHREAD_WRITE = 1, so the threads in all are nthread_compute + 2 (<= 8 with 6).
#
# The compute stage's time. FORCE times its three stages (input, compute, output) with time() and prints them in the
# end summary ("Virtual C-time"), but in whole seconds, too coarse for a stage of a second. force_time_ms.c is an
# LD_PRELOAD shim that makes time() count milliseconds (FORCE itself is not touched); the summary's fields are then
# decoded as milliseconds. The compute stage is screen_qai, the index, the TSA and the freeing of the inputs
# (H/tasks-hl.c compute_higher_level), with no reading and no writing. One run, one chunk (the whole tile).
#
# OMP_NUM_THREADS must be at least 4 and at least read + compute + write threads: FORCE runs the three stages as one
# OpenMP team, and with fewer it silently writes nothing ("data written 0.00 B"; measured: 1 compute thread needs
# OMP_NUM_THREADS=4, 3 and 2 fail). So the "OMP_NUM_THREADS = compute threads" of a plain setup does not work for 1.
# Inside the compute stage FORCE sets its own number of threads (NTHREAD_COMPUTE), so the extra OpenMP thread is idle.
#
# Output: $WORK/compute.tsv (config, nthread_compute, rep, wall_s, input_ms, compute_ms, output_ms), the outputs in
# $WORK/out/<config>/ and, in $DEST (the Windows folder force_tsa_timing.py reads: the series it feeds Zeit with, and
# the outputs of the accuracy check), the files it needs.
set -u
WORK=$(mkdir -p "${1:?workdir}" && cd "$1" && pwd)
NCOMP=${2:?nthread_compute}
shift 2
CONFIGS=("$@")
HERE=$(cd "$(dirname "$0")" && pwd)
DEST=${DEST:-/mnt/c/Users/eduar/AppData/Local/Temp/zeit-vs-force_tsa/bench}
SIZE=${SIZE:-500}
REPS=${REPS:-"1 2 3"}
# SENSOR=LND08 YEARS="2018 2019": a six-band Landsat cube (and the configurations spec_on, spec_off: FORCE's spectral
# adjustment to Sentinel-2A on and off, the ten adjusted bands or the six source bands as "indices") instead of the
# Sentinel-2A one; the cube's values go to $DEST/spec_src.npz for force_specadjust_timing.py. Use a workdir of its own.
SENSOR=${SENSOR:-SEN2A}
YEARS=${YEARS:-2018 2022}
YEAR0=${YEARS% *}; YEAR1=${YEARS#* }
source ~/force-ref-work/env.sh
OMP=$((NCOMP + 2)); [ $OMP -ge 4 ] || OMP=4   # see the note above: 3 is not enough, 4 is
export OMP_NUM_THREADS=$OMP GDAL_NUM_THREADS=1

STM="MIN Q01 Q10 Q25 Q50 Q75 Q90 Q99 MAX AVG STD RNG IQR SKW KRT NUM"
CSO="NUM MIN MAX RNG IQR AVG STD SKW KRT Q25 Q50 Q75"

mkdir -p "$WORK/prm" "$WORK/out" "$WORK/logs"
if [ ! -f "$WORK/ard/X0000_Y0000/.done" ]; then
    rm -rf "$WORK/ard"
    python -I "$HERE/force_tsa_cube.py" "$WORK" --size "$SIZE" --years $YEARS --seed 1 --sensor "$SENSOR" || exit 1
    touch "$WORK/ard/X0000_Y0000/.done"
fi
SHIM="$WORK/force_time_ms.so"
[ -f "$SHIM" ] && [ "$SHIM" -nt "$HERE/force_time_ms.c" ] || gcc -O2 -shared -fPIC -o "$SHIM" "$HERE/force_time_ms.c" || exit 1
TILE=$(awk '/^TILE_SIZE_X/ {printf "%d", $3}' "$WORK/ard/datacube-definition.prj")

[ -f "$WORK/tsa-template.prm" ] || force-parameter "$WORK/tsa-template.prm" TSA >/dev/null
[ -f "$WORK/cso-template.prm" ] || force-parameter "$WORK/cso-template.prm" CSO >/dev/null

# mkprm <template> <name> KEY=VALUE...: as in force_tsa.sh (a key missing from the template is an error).
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

# The summary's "Virtual X-time" fields, 00y 00m 00d 00h 00m 00s, as the milliseconds they count (progress-hl.c's
# set_secs: a year is 946080000, a month 2592000, a day 86400 of them).
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
        RESOLUTION=30 CHUNK_SIZE="$TILE $TILE" SENSORS="$SENSOR" TARGET_SENSOR=SEN2A DATE_RANGE="$YEAR0-01-01 $YEAR1-12-31")
# OUTPUT_TSS in every TSA run, so that all of them build the same TSS brick and the baseline's compute stage is what
# they have in common.
TSA_COMMON=(INDEX=NDVI OUTPUT_TSS=TRUE INT_DAY=16 ABOVE_NOISE=0 BELOW_NOISE=0)
RBF=(INTERPOLATE=RBF RBF_SIGMA="8 16 32" RBF_CUTOFF=0.95)
POL_ALL="DEM DLM DPS DSS DMS DES DEV DAV DLV LTS LGS LGV VEM VLM VPS VSS VMS VES VEV VAV VLV VBL VGA VSA VPA VGM VGV DPY DPV IST IBL IBT IGS IRR IFR RAR RAF RMR RMF"
FB=(OUTPUT_FBY=TRUE OUTPUT_FBQ=TRUE OUTPUT_FBM=TRUE OUTPUT_FBW=TRUE OUTPUT_FBD=TRUE)

[ -f "$WORK/compute.tsv" ] || printf 'config\tncomp\trep\twall_s\tinput_ms\tcompute_ms\toutput_ms\n' > "$WORK/compute.tsv"
FAILED=0

# cfg <name>: the module and the parameters of a configuration, in $MOD and $ARGS.
cfg() {
    MOD=TSA
    case $1 in
        baseline)        ARGS=(INTERPOLATE=NONE) ;;
        int_linear)      ARGS=(INTERPOLATE=LINEAR OUTPUT_TSI=TRUE) ;;
        int_moving)      ARGS=(INTERPOLATE=MOVING MOVING_MAX=16 OUTPUT_TSI=TRUE) ;;
        int_rbf)         ARGS=("${RBF[@]}" OUTPUT_TSI=TRUE) ;;
        int_harmonic)    ARGS=(INTERPOLATE=HARMONIC HARMONIC_TREND=TRUE HARMONIC_MODES=3
                               HARMONIC_FIT_RANGE="2018-01-01 2022-12-31" OUTPUT_TSI=TRUE) ;;
        rbf_stm)         ARGS=("${RBF[@]}" OUTPUT_STM=TRUE STM="$STM") ;;
        rbf_fold)        ARGS=("${RBF[@]}" FOLD_TYPE=AVG "${FB[@]}") ;;
        rbf_trend_nopen) ARGS=("${RBF[@]}" FOLD_TYPE=AVG OUTPUT_TRY=TRUE OUTPUT_CAY=TRUE CHANGE_PENALTY=FALSE) ;;
        rbf_polar)       ARGS=("${RBF[@]}" POL_ADAPTIVE=TRUE POL="VSS VPS VES VSA RMR IGS" OUTPUT_POL=TRUE) ;;
        rbf_polar39)     ARGS=("${RBF[@]}" POL_ADAPTIVE=TRUE POL="$POL_ALL" OUTPUT_POL=TRUE) ;;
        spec_on)         ARGS=(INDEX="BLUE GREEN RED REDEDGE1 REDEDGE2 REDEDGE3 BROADNIR NIR SWIR1 SWIR2" INTERPOLATE=NONE
                               SPECTRAL_ADJUST=TRUE) ;;
        spec_off)        ARGS=(INDEX="BLUE GREEN RED NIR SWIR1 SWIR2" INTERPOLATE=NONE SPECTRAL_ADJUST=FALSE
                               TARGET_SENSOR="$SENSOR") ;;
        tc)              ARGS=(INDEX="TC-BRIGHT TC-GREEN TC-WET TC-DI" INTERPOLATE=NONE TARGET_SENSOR="$SENSOR") ;;
        tc_src)          ARGS=(INDEX="BLUE GREEN RED NIR SWIR1 SWIR2" INTERPOLATE=NONE TARGET_SENSOR="$SENSOR") ;;
        cso_m1)          MOD=CSO; ARGS=(MONTH_STEP=1 CSO="$CSO") ;;
        cso_num)         MOD=CSO; ARGS=(MONTH_STEP=1 CSO="NUM") ;;
        *) echo "unknown config $1"; exit 2 ;;
    esac
}

# run <name> <rep>
run() {
    local name=$1 rep=$2
    cfg "$name"
    local tpl=$WORK/tsa-template.prm extra=()
    if [ "$MOD" = CSO ]; then tpl=$WORK/cso-template.prm; else extra=("${TSA_COMMON[@]}"); fi
    local o=$WORK/out/$name log=$WORK/logs/${name}_t${NCOMP}_r${rep}.log
    rm -rf "$o"; mkdir -p "$o"
    python -I "$WORK/mkprm.py" "$tpl" "$WORK/prm/$name.prm" "${COMMON[@]}" DIR_HIGHER="$o" DIR_PROVENANCE="$o" \
        "${extra[@]}" "${ARGS[@]}" || { echo "mkprm failed for $name"; FAILED=1; return; }
    local t0 t1 rc
    t0=$(date +%s.%N)
    LD_PRELOAD="$SHIM" force-higher-level "$WORK/prm/$name.prm" > "$log" 2>&1
    rc=$?
    t1=$(date +%s.%N)
    if [ $rc -ne 0 ] || grep -q "no input or output detected" "$log"; then
        echo "run $name failed ($rc), see $log"; FAILED=1; return
    fi
    local ms
    ms=$(python -I "$WORK/decode.py" "$log") || { FAILED=1; return; }
    printf '%s\t%d\t%d\t%.3f\t%s\n' "$name" "$NCOMP" "$rep" "$(echo "$t1 - $t0" | bc -l)" "${ms// /$'\t'}" >> "$WORK/compute.tsv"
    echo "done $name t=$NCOMP rep=$rep: wall $(echo "$t1 - $t0" | bc -l | cut -c1-6) s, I/C/O ms: $ms"
}

[ ${#CONFIGS[@]} -gt 0 ] || CONFIGS=(baseline int_linear int_moving int_rbf int_harmonic rbf_stm rbf_fold rbf_trend_nopen rbf_polar rbf_polar39 cso_m1 cso_num)
for name in "${CONFIGS[@]}"; do
    for rep in $REPS; do
        run "$name" "$rep"
    done
done

# The files force_tsa_timing.py needs: the TSS (baseline), the TSI of the four interpolations, the STM, the folds'
# FBY, the trend; the parameter files and compute.tsv. (Only for the configurations that exist.)
mkdir -p "$DEST/out"
for name in baseline int_linear int_moving int_rbf int_harmonic rbf_stm rbf_fold rbf_trend_nopen; do
    d=$WORK/out/$name/X0000_Y0000
    [ -d "$d" ] || continue
    mkdir -p "$DEST/out/$name/X0000_Y0000"
    case $name in
        baseline) pat="TSS" ;;
        int_*) pat="TSI" ;;
        rbf_stm) pat="STM" ;;
        rbf_fold) pat="FBY" ;;
        rbf_trend_nopen) pat="TRY CAY" ;;
    esac
    for p in $pat; do cp "$d"/*_${p}.tif "$DEST/out/$name/X0000_Y0000/"; done
done
[ -f "$WORK/spec_src.npz" ] && cp "$WORK/spec_src.npz" "$WORK/truth.npz" "$DEST/"
mkdir -p "$DEST/prm"
cp "$WORK"/prm/*.prm "$DEST/prm/"
cp "$WORK/compute.tsv" "$DEST/"
echo "copied to $DEST"
exit $FAILED
