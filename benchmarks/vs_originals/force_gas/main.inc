/* The harness itself (Zeit), appended to gas_harness.c by make_harness.sh after FORCE's functions.
 *
 *   gas_harness scene in.bin out.bin
 *     in.bin (little endian): int32 nb, nx, ny; float64 res; int32 gnx, gny; float64 gres; int32 nodata_view;
 *       int32 b_nir, b_vapor; int32 rsr_band[nb]; float32 cosszen[2], cosvzen[2]; float32 sun_czen[gn], view_czen[gn];
 *       float32 tso[nb][gn], tvo[nb][gn]; for z in 0..255: float32 rho_p[nb][gn], T[nb][gn], s[nb][gn];
 *       int16 toa[nb][nx*ny]; int16 qai[nx*ny]; uint8 dem[nx*ny]
 *     out.bin: int32 nlut; float32 lut[nlut] (nb*701*101); int16 wvp[nx*ny]; int16 qai[nx*ny] (after water_vapor);
 *       int16 tg[nb][nx*ny]. Prints the time of water_vapor and gas_transmittance (ms).
 *   gas_harness lut < scenarios.txt
 *     one line each: dir lon lat year month day wvp strict   (dir NULL: the constant wvp); prints the value (%.9g;
 *     -1 where FORCE returns -1, with its message, if any, on a line of its own). */
static double now_ms(void){
  struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}

static void rd(void *p, size_t sz, size_t n, FILE *f){
  if (fread(p, sz, n, f) != n){ fprintf(stderr, "short read\n"); exit(2); }
}

static brick_t *mkbrick(int nx, int ny, double res, int nb, int nodata, int kind){
  brick_t *b = calloc(1, sizeof(brick_t));
  b->nx = nx; b->ny = ny; b->nb = nb; b->geotran[_GT_RES_] = res;
  for (int k = 0; k < 8; k++) b->nodata[k] = nodata;
  if (kind == 0){ b->vfloat = calloc(nb, sizeof(float*)); for (int k = 0; k < nb; k++) b->vfloat[k] = calloc((size_t)nx * ny, sizeof(float)); }
  if (kind == 1){ b->vshort = calloc(nb, sizeof(short*)); for (int k = 0; k < nb; k++) b->vshort[k] = calloc((size_t)nx * ny, sizeof(short)); }
  if (kind == 2){ b->vsmall = calloc(nb, sizeof(small*)); for (int k = 0; k < nb; k++) b->vsmall[k] = calloc((size_t)nx * ny, 1); }
  return b;
}

static int scene(const char *fin, const char *fout){
  FILE *f = fopen(fin, "rb");
  if (!f){ perror("input"); return 2; }
  int nb, nx, ny, gnx, gny, nodata, b_nir, b_vapor;
  double res, gres;
  rd(&nb, 4, 1, f); rd(&nx, 4, 1, f); rd(&ny, 4, 1, f); rd(&res, 8, 1, f);
  rd(&gnx, 4, 1, f); rd(&gny, 4, 1, f); rd(&gres, 8, 1, f); rd(&nodata, 4, 1, f);
  rd(&b_nir, 4, 1, f); rd(&b_vapor, 4, 1, f);
  G_B_NIR = b_nir; G_B_VAPOR = b_vapor;
  int gn = gnx * gny, npix = nx * ny;
  meta_t meta; meta.cal = calloc(nb, sizeof(cal_t));
  for (int b = 0; b < nb; b++){ int r; rd(&r, 4, 1, f); meta.cal[b].rsr_band = r; }
  atc_t atc;
  rd(atc.cosszen, 4, 2, f); rd(atc.cosvzen, 4, 2, f);
  atc.xy_sun = mkbrick(gnx, gny, gres, 8, nodata, 0);
  atc.xy_view = mkbrick(gnx, gny, gres, 8, nodata, 0);
  atc.xy_Tso = mkbrick(gnx, gny, gres, nb, nodata, 0);
  atc.xy_Tvo = mkbrick(gnx, gny, gres, nb, nodata, 0);
  atc.xy_Tg = mkbrick(gnx, gny, gres, nb, nodata, 0);
  rd(atc.xy_sun->vfloat[cZEN], 4, gn, f); rd(atc.xy_view->vfloat[cZEN], 4, gn, f);
  for (int b = 0; b < nb; b++) rd(atc.xy_Tso->vfloat[b], 4, gn, f);
  for (int b = 0; b < nb; b++) rd(atc.xy_Tvo->vfloat[b], 4, gn, f);
  atc.xyz_rho_p = calloc(_BYTE_LEN_, sizeof(brick_t*));
  atc.xyz_T = calloc(_BYTE_LEN_, sizeof(brick_t*));
  atc.xyz_s = calloc(_BYTE_LEN_, sizeof(brick_t*));
  for (int z = 0; z < _BYTE_LEN_; z++){
    atc.xyz_rho_p[z] = mkbrick(gnx, gny, gres, nb, nodata, 0);
    atc.xyz_T[z] = mkbrick(gnx, gny, gres, nb, nodata, 0);
    atc.xyz_s[z] = mkbrick(gnx, gny, gres, nb, nodata, 0);
    for (int b = 0; b < nb; b++) rd(atc.xyz_rho_p[z]->vfloat[b], 4, gn, f);
    for (int b = 0; b < nb; b++) rd(atc.xyz_T[z]->vfloat[b], 4, gn, f);
    for (int b = 0; b < nb; b++) rd(atc.xyz_s[z]->vfloat[b], 4, gn, f);
  }
  brick_t *TOA = mkbrick(nx, ny, res, nb, -9999, 1);
  brick_t *QAI = mkbrick(nx, ny, res, 1, -9999, 1);
  brick_t *DEM = mkbrick(nx, ny, res, 1, -9999, 2);
  for (int b = 0; b < nb; b++) rd(TOA->vshort[b], 2, npix, f);
  rd(QAI->vshort[0], 2, npix, f);
  rd(DEM->vsmall[0], 1, npix, f);
  fclose(f);

  double t0 = now_ms();
  brick_t *WVP = water_vapor(&meta, &atc, TOA, QAI, DEM);
  double t1 = now_ms();
  if (WVP == NULL){ fprintf(stderr, "water_vapor failed\n"); return 3; }
  f = fopen(fout, "wb");
  int nlut = _WVLUT_.nb * _WVLUT_.nw * _WVLUT_.nm;
  fwrite(&nlut, 4, 1, f);
  for (int b = 0; b < _WVLUT_.nb; b++) for (int kw = 0; kw < _WVLUT_.nw; kw++)
    fwrite(_WVLUT_.val[b][kw], 4, _WVLUT_.nm, f);
  fwrite(WVP->vshort[0], 2, npix, f);
  fwrite(QAI->vshort[0], 2, npix, f);
  double t2 = now_ms();
  for (int b = 0; b < nb; b++){
    short *tg = gas_transmittance(&atc, b, WVP, QAI);
    fwrite(tg, 2, npix, f);
    free(tg);
  }
  double t3 = now_ms();
  fclose(f);
  printf("%.3f %.3f\n", t1 - t0, t3 - t2);
  free_wvlut();
  return 0;
}

static int lut(void){
  char dir[NPOW_10];
  double lon, lat, wvp; int year, month, day, strict;
  while (scanf("%1023s %lf %lf %d %d %d %lf %d", dir, &lon, &lat, &year, &month, &day, &wvp, &strict) == 8){
    par_ll_t pl2;
    pl2.d_wvp = dir; pl2.wvp = (float)wvp; pl2.wvp_strict = strict;
    atc_t atc;
    atc.xy_sun = calloc(1, sizeof(brick_t));
    atc.xy_sun->nx = 2; atc.xy_sun->ny = 2;
    atc.xy_sun->lon = lon; atc.xy_sun->lat = lat;
    atc.xy_sun->year = year; atc.xy_sun->month = month; atc.xy_sun->day = day;
    printf("%.9g\n", water_vapor_from_lut(&pl2, &atc));
    free(atc.xy_sun);
  }
  return 0;
}

int main(int argc, char **argv){
  if (argc >= 4 && !strcmp(argv[1], "scene")) return scene(argv[2], argv[3]);
  if (argc >= 2 && !strcmp(argv[1], "lut")) return lut();
  fprintf(stderr, "usage: %s scene in.bin out.bin | lut < scenarios.txt\n", argv[0]);
  return 2;
}
