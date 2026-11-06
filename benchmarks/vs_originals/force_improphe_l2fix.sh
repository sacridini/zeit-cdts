#!/bin/bash
# Builds the one binary the L2-ImproPhe comparison needs: FORCE 3.11.00's force-higher-level with the secondary sensors'
# settings filled in. In 3.11.00 only `phl->sen` is registered with the parameter parser (TARGET_SENSOR, PRODUCT_TYPE_MAIN,
# PRODUCT_TYPE_QUALITY); `phl->sen2`, the SENSORS_LOWRES of L2IMP, keeps NULL for its target, main and quality product, so
# retrieve_sensor(&phl->sen2) stops at "Could not parse sensor definition for target sensor (null)" and, past that, the
# reading of the secondary scenes (strstr on a NULL main_product) segfaults: L2IMP does not run as released. The copy of
# the build tree (the original tree and the installed binaries are not touched) gets, before that call in param-hl.c,
#
#     sen2.target = "LNDLG" (the six bands of Landsat 8, which check_target_sensor compares the band count with),
#     sen2.main_product and sen2.quality_product = copies of the primary sensors' (BOA and QAI by default);
#
# param-hl.o is rebuilt with the Makefile's flags and force-higher-level relinked with the Makefile's own link command.
# Nothing of the improphe code differs. Usage (WSL): bash force_improphe_l2fix.sh   (one core)
set -eu
source ~/force-ref-work/env.sh
SRC=${SRC:-$HOME/force-src}
DST=${DST:-$HOME/force-ref-work/force-l2fix}
rm -rf "$DST"; cp -a "$SRC" "$DST"; cd "$DST"
python - <<'PY'
p = "src/modules/higher-level/param-hl.c"
s = open(p).read()
old = """  if ((phl->input_level2 == _INP_QAI_ ||
       phl->input_level2 == _INP_ARD_) &&
    retrieve_sensor(&phl->sen2) != SUCCESS){"""
new = """  if (phl->sen2.target == NULL){
    alloc((void**)&phl->sen2.target, NPOW_10, sizeof(char));
    copy_string(phl->sen2.target, NPOW_10, "LNDLG");
  }
  if (phl->sen2.main_product == NULL){
    alloc((void**)&phl->sen2.main_product, NPOW_10, sizeof(char));
    copy_string(phl->sen2.main_product, NPOW_10, phl->sen.main_product);
  }
  if (phl->sen2.quality_product == NULL){
    alloc((void**)&phl->sen2.quality_product, NPOW_10, sizeof(char));
    copy_string(phl->sen2.quality_product, NPOW_10, phl->sen.quality_product);
  }
""" + old
assert s.count(old) == 1
open(p, "w").write(s.replace(old, new))
PY
make -n exe > /tmp/make_n.txt 2>&1 || true
python - <<'PY'
lines = open("/tmp/make_n.txt").read().split("\n")
for i, l in enumerate(lines):
    if "-o bin/force-higher-level " in l:
        out = [l]
        while out[-1].endswith("\\"):
            i += 1
            out.append(lines[i])
        open("/tmp/hl_link.sh", "w").write("\n".join(out) + "\n")
        break
PY
gcc -O3 -Wall -fopenmp -c src/modules/higher-level/param-hl.c -o obj/param-hl.o
bash /tmp/hl_link.sh
ls -la bin/force-higher-level
