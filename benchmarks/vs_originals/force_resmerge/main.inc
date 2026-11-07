
/* ---- the harness itself (Zeit) ------------------------------------------------------------------------------
 * resmerge_harness <scene.bin> <out.bin> <method 1|2>
 * scene.bin: int32 nx, int32 ny, then int16 bands [9][ny*nx] (green, red, broad NIR, red edge 1-3, NIR (B8A),
 * SWIR1, SWIR2) and bytes off, cloud, shadow [ny*nx] each. out.bin: the six predicted bands, int16 [6][ny*nx].
 * Prints the time of the call (ms). The number of threads is OMP_NUM_THREADS. */
int main(int argc, char **argv){
  if (argc < 4){ fprintf(stderr, "usage: %s scene.bin out.bin method\n", argv[0]); return 2; }
  FILE *f = fopen(argv[1], "rb");
  if (!f){ perror("scene"); return 2; }
  int nx, ny;
  if (fread(&nx, 4, 1, f) != 1 || fread(&ny, 4, 1, f) != 1) return 2;
  size_t n = (size_t)nx * ny;
  brick_t b;
  b.nx = nx; b.ny = ny;
  b.toa = malloc(9 * sizeof(short*));
  for (int k = 0; k < 9; k++){
    b.toa[k] = malloc(n * sizeof(short));
    if (fread(b.toa[k], 2, n, f) != n) return 2;
  }
  b.off = malloc(n); b.cloud = malloc(n); b.shadow = malloc(n);
  if (fread(b.off, 1, n, f) != n || fread(b.cloud, 1, n, f) != n || fread(b.shadow, 1, n, f) != n) return 2;
  fclose(f);

  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  int err = atoi(argv[3]) == 1 ? resolution_merge_1(&b, &b) : resolution_merge_2(&b, &b);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  if (err != SUCCESS){ fprintf(stderr, "merge failed\n"); return 1; }
  printf("%.3f\n", (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) * 1e-6);

  f = fopen(argv[2], "wb");
  for (int k = 3; k < 9; k++) fwrite(b.toa[k], 2, n, f);
  fclose(f);
  return 0;
}
