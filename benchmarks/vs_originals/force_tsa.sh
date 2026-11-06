#!/bin/bash
# The FORCE side of the TSA/CSO comparison: generates the synthetic ARD cube (force_tsa_cube.py), runs FORCE 3.11.00's
# higher-level TSA and CSO on it, one configuration per run (each in its own output folder), times every run and
# copies the small outputs to the Windows folder the comparison scripts read (workdir("force_tsa")/ref).
#
#   wsl -e bash -lc 'bash /mnt/c/.../benchmarks/vs_originals/force_tsa.sh ~/force-ref-work/tsa-ref 6'
#
# Usage: force_tsa.sh <workdir> [nthread_compute=6]. NTHREAD_READ = NTHREAD_WRITE = 1, so the total stays <= 8.
# The cube (ard/, BOA and QAI) and the outputs (out/<config>/) stay in <workdir> (the WSL disk; /mnt/c is slow); the
# outputs, the parameter files, times.tsv and truth.npz are copied to $DEST. One run at a time, never in parallel.
set -u
WORK=$(mkdir -p "${1:?workdir}" && cd "$1" && pwd)
NCOMP=${2:-6}
HERE=$(cd "$(dirname "$0")" && pwd)
DEST=${DEST:-/mnt/c/Users/eduar/AppData/Local/Temp/zeit-vs-force_tsa/ref}
SIZE=${SIZE:-100}
source ~/force-ref-work/env.sh
# OMP (default NCOMP) is the OpenMP team's size; FORCE runs reading, compute and writing as one team, which must have
# at least read + compute + write threads or nothing is written (see force_tsa_timing.sh): OMP=8 with NCOMP=6.
export OMP_NUM_THREADS=${OMP:-$NCOMP} GDAL_NUM_THREADS=1
# ONLY="name name": run (and copy) just those configurations, leaving the other outputs in $DEST as they are.
ONLY=${ONLY:-}

STM="MIN Q01 Q10 Q25 Q50 Q75 Q90 Q99 MAX AVG STD RNG IQR SKW KRT NUM"
CSO="NUM MIN MAX RNG IQR AVG STD SKW KRT Q25 Q50 Q75"

mkdir -p "$WORK/prm" "$WORK/out" "$WORK/logs"
if [ ! -f "$WORK/ard/X0000_Y0000/.done" ]; then
    rm -rf "$WORK/ard"
    python -I "$HERE/force_tsa_cube.py" "$WORK" --size "$SIZE" --years 2018 2022 --seed 1 || exit 1
    touch "$WORK/ard/X0000_Y0000/.done"
fi
TILE=$(awk '/^TILE_SIZE_X/ {printf "%d", $3}' "$WORK/ard/datacube-definition.prj")

[ -f "$WORK/tsa-template.prm" ] || force-parameter "$WORK/tsa-template.prm" TSA >/dev/null
[ -f "$WORK/cso-template.prm" ] || force-parameter "$WORK/cso-template.prm" CSO >/dev/null

# mkprm <template> <name> KEY=VALUE...: a copy of the template with the common settings and the given ones; every key
# must exist in the template (a typo is an error, not a silently ignored line).
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
        RESOLUTION=30 CHUNK_SIZE="$TILE $TILE" SENSORS=SEN2A TARGET_SENSOR=SEN2A DATE_RANGE="2018-01-01 2022-12-31")
# The noise filter is off (ABOVE_NOISE = BELOW_NOISE = 0 cancels it, H/quality-hl.c:178), so the TSS keeps every
# observation the QAI lets through; the CSO has no such keys.
TSA_COMMON=(INDEX=NDVI OUTPUT_TSS=TRUE INT_DAY=16 ABOVE_NOISE=0 BELOW_NOISE=0)

TIMES=$WORK/times.tsv
[ -n "$ONLY" ] && TIMES=$WORK/times_only.tsv
: > "$TIMES"
printf 'config\tseconds\texit\n' >> "$TIMES"
FAILED=0

# run <TSA|CSO> <name> KEY=VALUE...
run() {
    local mod=$1 name=$2; shift 2
    if [ -n "$ONLY" ]; then
        case " $ONLY " in *" $name "*) ;; *) return ;; esac
    fi
    local tpl=$WORK/tsa-template.prm
    [ "$mod" = CSO ] && tpl=$WORK/cso-template.prm
    local o=$WORK/out/$name
    rm -rf "$o"; mkdir -p "$o"
    local extra=()
    [ "$mod" = TSA ] && extra=("${TSA_COMMON[@]}")
    python -I "$WORK/mkprm.py" "$tpl" "$WORK/prm/$name.prm" "${COMMON[@]}" DIR_HIGHER="$o" DIR_PROVENANCE="$o" \
        "${extra[@]}" "$@" || { echo "mkprm failed for $name"; FAILED=1; return; }
    local t0 t1 rc
    t0=$(date +%s.%N)
    force-higher-level "$WORK/prm/$name.prm" > "$WORK/logs/$name.log" 2>&1
    rc=$?
    t1=$(date +%s.%N)
    printf '%s\t%.3f\t%d\n' "$name" "$(echo "$t1 - $t0" | bc -l)" "$rc" >> "$TIMES"
    [ $rc -eq 0 ] || { echo "run $name failed ($rc), see $WORK/logs/$name.log"; FAILED=1; }
    echo "done $name ($rc)"
}

# Baseline: no interpolation, only the TSS, to take the reading and the writing off the other times.
run TSA baseline INTERPOLATE=NONE

# Interpolation (the TSS and the TSI written).
run TSA int_none     INTERPOLATE=NONE OUTPUT_TSI=TRUE
run TSA int_linear   INTERPOLATE=LINEAR OUTPUT_TSI=TRUE
run TSA int_moving   INTERPOLATE=MOVING MOVING_MAX=16 OUTPUT_TSI=TRUE
run TSA int_rbf      INTERPOLATE=RBF RBF_SIGMA="8 16 32" RBF_CUTOFF=0.95 OUTPUT_TSI=TRUE
run TSA int_harmonic INTERPOLATE=HARMONIC HARMONIC_TREND=TRUE HARMONIC_MODES=3 \
    HARMONIC_FIT_RANGE="2018-01-01 2022-12-31" OUTPUT_TSI=TRUE
run TSA int_harmonic_part INTERPOLATE=HARMONIC HARMONIC_TREND=TRUE HARMONIC_MODES=3 \
    HARMONIC_FIT_RANGE="2018-01-01 2020-12-31" OUTPUT_TSI=TRUE OUTPUT_NRT=TRUE

RBF=(INTERPOLATE=RBF RBF_SIGMA="8 16 32" RBF_CUTOFF=0.95)
FB=(OUTPUT_FBY=TRUE OUTPUT_FBQ=TRUE OUTPUT_FBM=TRUE OUTPUT_FBW=TRUE OUTPUT_FBD=TRUE)

# Over the RBF: the STM, the folds, the trend and the change/trend analysis (with and without the penalty).
run TSA rbf_stm "${RBF[@]}" OUTPUT_STM=TRUE STM="$STM"
run TSA rbf_fold "${RBF[@]}" FOLD_TYPE=AVG "${FB[@]}"
run TSA rbf_trend_nopen "${RBF[@]}" FOLD_TYPE=AVG OUTPUT_TRY=TRUE OUTPUT_TRM=TRUE OUTPUT_CAY=TRUE OUTPUT_CAM=TRUE \
    CHANGE_PENALTY=FALSE
run TSA rbf_trend_pen "${RBF[@]}" FOLD_TYPE=AVG OUTPUT_TRY=TRUE OUTPUT_TRM=TRUE OUTPUT_CAY=TRUE OUTPUT_CAM=TRUE \
    CHANGE_PENALTY=TRUE

# One FOLD_TYPE per run; only the folds on.
for ft in AVG Q50 STD MIN MAX SKW KRT IQR Q25 NUM; do
    run TSA "fold_$ft" "${RBF[@]}" FOLD_TYPE=$ft "${FB[@]}"
done

# Standardization.
run TSA standardize "${RBF[@]}" OUTPUT_TSI=TRUE STANDARDIZE_TSI=NORMALIZE STANDARDIZE_FOLD=CENTER FOLD_TYPE=AVG \
    OUTPUT_FBY=TRUE OUTPUT_FBQ=TRUE OUTPUT_FBM=TRUE

# CSO.
run CSO cso_m1 MONTH_STEP=1 CSO="$CSO"
run CSO cso_m3 MONTH_STEP=3 CSO="$CSO"

# Polar phenology metrics (FORCE's POL, with the trend and the change of each: TRO, CAO), over the RBF: the 39 metrics, with
# the start of each year's season fine-tuned from its own mean vector (adaptive) and from the long-term one (fixed).
POL_ALL="DEM DLM DPS DSS DMS DES DEV DAV DLV LTS LGS LGV VEM VLM VPS VSS VMS VES VEV VAV VLV VBL VGA VSA VPA VGM VGV \
DPY DPV IST IBL IBT IGS IRR IFR RAR RAF RMR RMF"
for adaptive in adapt fixed; do
    ad=TRUE; [ $adaptive = fixed ] && ad=FALSE
    run TSA "rbf_polar_$adaptive" "${RBF[@]}" POL_ADAPTIVE=$ad POL="$POL_ALL" OUTPUT_POL=TRUE OUTPUT_TRO=TRUE \
        OUTPUT_CAO=TRUE
done

# Copy the small outputs (the cube's BOA and QAI stay here).
mkdir -p "$DEST"
if [ -n "$ONLY" ]; then
    mkdir -p "$DEST/out" "$DEST/prm"
    for name in $ONLY; do
        rm -rf "$DEST/out/$name" "$DEST/prm/$name.prm"
        cp -r "$WORK/out/$name" "$DEST/out/"
        cp "$WORK/prm/$name.prm" "$DEST/prm/"
    done
    cp "$WORK/times_only.tsv" "$DEST/"
else
    rm -rf "$DEST/out" "$DEST/prm"
    cp -r "$WORK/out" "$WORK/prm" "$DEST/"
    cp "$WORK/times.tsv" "$WORK/truth.npz" "$DEST/"
fi
echo "copied to $DEST"
cat "$TIMES"
exit $FAILED
