
/* ---- the harness itself (Zeit) ------------------------------------------------------------------------------
 * parallax_harness parallax <scene.bin> <out.bin> <cdi.bin>
 *   scene.bin: int32 nx, int32 ny, int16 re3, bnir, nir, var (x 10000) [ny*nx] each, then bytes pcp, off, brt, sat
 *   [ny*nx] each. out.bin: the cloud mask (bytes, ny*nx); cdi.bin: the CDI (float, (ny/2)*(nx/2)).
 * parallax_harness buffer <in.bin> <out.bin> <radius>
 *   in.bin: int32 nx, int32 ny, bytes [ny*nx]; out.bin: bytes.
 * Prints the time of the call (ms). The number of threads is OMP_NUM_THREADS. */
static double now_ms(void){
  struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}

int main(int argc, char **argv){
  if (argc < 5){ fprintf(stderr, "usage: %s parallax scene out cdi | buffer in out r\n", argv[0]); return 2; }
  FILE *f = fopen(argv[2], "rb");
  if (!f){ perror("input"); return 2; }
  int nx, ny;
  if (fread(&nx, 4, 1, f) != 1 || fread(&ny, 4, 1, f) != 1) return 2;
  size_t n = (size_t)nx * ny;
  if (!strcmp(argv[1], "buffer")){
    small *img = malloc(n);
    if (fread(img, 1, n, f) != n) return 2;
    fclose(f);
    double t0 = now_ms();
    int err = buffer_(img, nx, ny, atoi(argv[4]));
    double t1 = now_ms();
    if (err != SUCCESS) return 1;
    printf("%.3f\n", t1 - t0);
    f = fopen(argv[3], "wb"); fwrite(img, 1, n, f); fclose(f);
    return 0;
  }
  brick_t b;
  b.nx = nx; b.ny = ny;
  short *var = malloc(n * 2);
  small *pcp = malloc(n), *brt = malloc(n), *clr = calloc(n, 1), *lnd = calloc(n, 1);
  b.re3 = malloc(n * 2); b.bnir = malloc(n * 2); b.nir = malloc(n * 2);
  b.off = malloc(n); b.sat = malloc(n);
  if (fread(b.re3, 2, n, f) != n || fread(b.bnir, 2, n, f) != n || fread(b.nir, 2, n, f) != n ||
      fread(var, 2, n, f) != n || fread(pcp, 1, n, f) != n || fread(b.off, 1, n, f) != n ||
      fread(brt, 1, n, f) != n || fread(b.sat, 1, n, f) != n) return 2;
  fclose(f);

  small *cld = NULL;
  int ncloud = 0;
  float cc = 0;
  double t0 = now_ms();
  int err = cloud_parallax(0, 0, (int)n, &ncloud, &cc, &b, &b, pcp, clr, lnd, brt, var, &cld);
  double t1 = now_ms();
  if (err != SUCCESS){ fprintf(stderr, "cloud_parallax failed\n"); return 1; }
  printf("%.3f\n", t1 - t0);
  f = fopen(argv[3], "wb"); fwrite(cld, 1, n, f); fclose(f);
  f = fopen(argv[4], "wb"); fwrite(dumped_cdi, 4, (size_t)(ny / 2) * (nx / 2), f); fclose(f);
  return 0;
}
