#!/bin/bash
# Builds a copy of FORCE 3.11.00's force-higher-level that also dumps the intermediate arrays of CF-ImproPhe's PCA to binary
# files, to find the first stage at which Zeit and FORCE differ (force_improphe.py reports only the end result). A copy of the
# build tree (the original tree and the installed binaries are not touched), with the secondary-sensor fix of
# force_improphe_l2fix.sh. The dumps go to the folder $FORCE_DUMP (an environment variable of the run), one set per call of
# pca(): pca_<k>_in.i16 (the seasonal means, (nb, nc) int16), _cov.f64, _eval.f64 and _evec.f64 (after the sort, as
# GSL's row-major matrix, columns = eigenvectors), _pc.f32 ((numcomp, nc), before the standardization), and, from
# cf-improphe-hl.c, cfi_<y>_std.f32 (the standardized PCs) and cfi_<y>_tex.f32 (the texture). Nothing else changes.
# Usage (WSL): bash force_improphe_dump.sh   (one core); run with FORCE_DUMP=/some/dir force-dbg-hl prm.
set -eu
source ~/force-ref-work/env.sh
SRC=${SRC:-$HOME/force-ref-work/force-l2fix}
DST=${DST:-$HOME/force-ref-work/force-dump}
rm -rf "$DST"; cp -a "$SRC" "$DST"; cd "$DST"
python - <<'PY'
p = "src/modules/cross-level/pca-cl.c"
s = open(p).read()
s = s.replace("#include", "#include <stdlib.h>\n#include <stdio.h>\n#include", 1)
s = s.replace("""float **pca(short **INP""", """static void dump_file(const char *name, int k, const char *ext, const void *data, size_t size, size_t n){
  const char *dir = getenv("FORCE_DUMP");
  if (dir == NULL) return;
  char path[4096];
  snprintf(path, sizeof(path), "%s/%s_%d_%s", dir, name, k, ext);
  FILE *fp = fopen(path, "ab");   // appended: pca() dumps one band at a time; empty the folder before a run
  if (fp == NULL) return;
  fwrite(data, size, n, fp);
  fclose(fp);
}

float **pca(short **INP""", 1)
s = s.replace("int numcomp = nb;", "int numcomp = nb;\nstatic int call = 0;\nint kcall = call++;", 1)
s = s.replace("""  free((void*)mean);
""", """  free((void*)mean);
  for (b=0; b<nb; b++) dump_file("pca", kcall, "in.i16", INP[b], sizeof(short), nc);
""", 1)
s = s.replace("""  // find eigen-values and eigen-vectors""", """  dump_file("pca", kcall, "cov.f64", covm->data, sizeof(double), (size_t)nb*nb);
  // find eigen-values and eigen-vectors""", 1)
s = s.replace("""  // find how many components to keep""", """  dump_file("pca", kcall, "eval.f64", eval->data, sizeof(double), nb);
  dump_file("pca", kcall, "evec.f64", evec->data, sizeof(double), (size_t)nb*nb);
  // find how many components to keep""", 1)
s = s.replace("""  // clean
  gsl_vector_free(eval);""", """  for (b=0; b<numcomp; b++) dump_file("pca", kcall, "pc.f32", PCA[b], sizeof(float), nc);
  // clean
  gsl_vector_free(eval);""", 1)
open(p, "w").write(s)

p = "src/modules/higher-level/cf-improphe-hl.c"
s = open(p).read()
s = s.replace("""    // HR texture
    if ((ard_tex_ = focal_sd(ard_, ard_nodata, phl->imp.ksd, nx, ny, npc, 0)) == NULL){
      printf("error in computing high-res texture.\\n");
      *nproduct = 0;
      return NULL;}
""", """    // HR texture
    if ((ard_tex_ = focal_sd(ard_, ard_nodata, phl->imp.ksd, nx, ny, npc, 0)) == NULL){
      printf("error in computing high-res texture.\\n");
      *nproduct = 0;
      return NULL;}
    if (getenv("FORCE_DUMP") != NULL){
      char path[4096]; FILE *fp;
      for (b=0; b<npc; b++){
        snprintf(path, sizeof(path), "%s/cfi_%d_std.f32", getenv("FORCE_DUMP"), y);
        fp = fopen(path, b == 0 ? "wb" : "ab"); fwrite(ard_[b], sizeof(float), nc, fp); fclose(fp);
      }
      snprintf(path, sizeof(path), "%s/cfi_%d_tex.f32", getenv("FORCE_DUMP"), y);
      fp = fopen(path, "wb"); fwrite(ard_tex_, sizeof(float), nc, fp); fclose(fp);
    }
""", 1)
assert "cfi_%d_tex" in s
open(p, "w").write(s)
PY
grep -q "dump_file" src/modules/cross-level/pca-cl.c
make -n exe > /tmp/make_n_dump.txt 2>&1 || true
python - <<'PY'
lines = open("/tmp/make_n_dump.txt").read().split("\n")
for i, l in enumerate(lines):
    if "-o bin/force-higher-level " in l:
        out = [l]
        while out[-1].endswith("\\"):
            i += 1
            out.append(lines[i])
        open("/tmp/hl_link_dump.sh", "w").write("\n".join(out) + "\n")
        break
PY
# the Makefile's own compile commands of the two patched files
grep -E -- "-c src/modules/cross-level/pca-cl.c -o|-c src/modules/higher-level/cf-improphe-hl.c -o" /tmp/make_n_dump.txt > /tmp/compile_dump.sh
test "$(wc -l < /tmp/compile_dump.sh)" = 2
bash /tmp/compile_dump.sh
bash /tmp/hl_link_dump.sh
ls -la bin/force-higher-level
