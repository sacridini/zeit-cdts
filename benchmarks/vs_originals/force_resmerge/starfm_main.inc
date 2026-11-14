
/* ---- the harness itself (Zeit) ------------------------------------------------------------------------------
 * starfm_harness <in.bin> <out.bin> <radius> [nclass unc_coarse unc_fine]
 * in.bin: int32 nx, ny, npair, ntarget; float32 fine [npair][ny*nx]; float32 coarse [npair+ntarget][ny*nx] (the
 * pairs' coarse images, then the images at the prediction dates); bytes off [ntarget][ny*nx].
 * out.bin: float32 [ntarget][ny*nx], FORCE's starfm() once per target (0 where off). */
int main(int argc, char **argv){
  if (argc < 4){ fprintf(stderr, "usage: %s in.bin out.bin radius [nclass unc_coarse unc_fine]\n", argv[0]); return 2; }
  FILE *f = fopen(argv[1], "rb");
  if (!f){ perror("input"); return 2; }
  int nx, ny, npair, nt;
  if (fread(&nx, 4, 1, f) != 1 || fread(&ny, 4, 1, f) != 1 || fread(&npair, 4, 1, f) != 1 || fread(&nt, 4, 1, f) != 1) return 2;
  size_t n = (size_t)nx * ny;
  if (npair < 1 || npair > 16) return 2;
  int r = atoi(argv[3]);
  g_npair = npair;
  if (argc > 4) g_nclass = (float)atof(argv[4]);
  if (argc > 5) g_unc_coarse = (float)atof(argv[5]);
  if (argc > 6) g_unc_fine = (float)atof(argv[6]);
  float **fine = malloc(npair * sizeof(float*));
  float **coarse = malloc((npair + 1) * sizeof(float*));
  for (int k = 0; k < npair; k++){ fine[k] = malloc(n * 4); if (fread(fine[k], 4, n, f) != n) return 2; }
  float **all = malloc((npair + nt) * sizeof(float*));
  for (int k = 0; k < npair + nt; k++){ all[k] = malloc(n * 4); if (fread(all[k], 4, n, f) != n) return 2; }
  brick_t b;
  b.nx = nx; b.ny = ny; b.toa = NULL; b.cloud = b.shadow = NULL;
  b.off = malloc(n);
  FILE *o = fopen(argv[2], "wb");
  for (int t = 0; t < nt; t++){
    if (fread(b.off, 1, n, f) != n) return 2;
    for (int k = 0; k < npair; k++) coarse[k] = all[k];
    coarse[npair] = all[npair + t];
    float *pred = starfm(nx, ny, &b, coarse, fine, r);
    fwrite(pred, 4, n, o);
    free(pred);
  }
  fclose(o);
  fclose(f);
  return 0;
}
