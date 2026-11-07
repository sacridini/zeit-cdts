/* The harness itself (Zeit), appended to aod_harness.c by make_harness.sh after FORCE's functions.
 *
 *   aod_harness scene in.bin out.bin
 *     in.bin (little endian): int32 nb, nx, ny; float64 res; int32 gnx, gny; float64 gres; int32 nodata_view, multi;
 *       char domain[nb][16]; int32 rsr_band[nb]; uint8 aod_bands[nb]; float32 wvl[nb], lwvl[nb], E0[nb];
 *       float32 dem_avg, dem_min, dem_max, dem_step, sob;
 *       coarse grid (gn = gnx*gny): float32 sun_czen[gn], sun_azi[gn], view_czen[gn], view_azi[gn], tso[nb][gn],
 *       tvo[nb][gn], tg[nb][gn], mod[nb][gn], brdf[nb][gn], pr[gn], pa[gn], hr[gn], fresnel[gn], dem_bin[gn];
 *       pixels (npix = nx*ny): int16 toa[nb][npix], qai[npix]; uint8 top_dem[npix]; uint16 slope[npix]; int16 ill[npix];
 *       uint16 sky[npix].
 *     out.bin: the order of compile_aod up to the AOD map, with the dumps of the stages: extract_dark_target of the
 *       water and the vegetation type (an object: int32 k, valid, imin, imax, jmin, jmax, i, j, e, f, g, r; float32 ms,
 *       mv, razi, cosi, sky, z, Hr; float32 ttoa[nb], etoa[nb]; preceded by int32 num); aod_lib_to_target on every valid
 *       target of those (int32 nvalid; per target float32 aodest[lib.n][nb]); aod_from_target of both (int32 k; per
 *       object int32 valid, lib_id, ang_fit; float32 rsq, coef[3], aod[nb], est[nb]); aod_elev_dependency (float32 Hp,
 *       Ha; xy_Ha[gn]; the Ha of each water and vegetation object). Prints the times of the stages (ms).
 *   aod_harness compile in.bin out.bin doaod dir lon lat doy: FORCE's compile_aod as it is on the same scene (dir "NULL": the
 *     global fallback): int32 ret, aodmap; float32 aod[nb], Hp, Ha; int32 aod_bands[nb]; float32 xy_aod[nb][gn], xy_interp[gn],
 *     xy_Ha[gn]; prints the time (ms).
 *   aod_harness idw in.bin out.bin: GDALGridCreate as FORCE's interpolate_aod_map calls it (see idw()).
 *   aod_harness image in.bin out.bin: int32 nx, ny, uint8 image[nx*ny] -> int32 n, uint16 dist, int32 labels.
 *   aod_harness elev < cases.txt: aod_elev_dependency on small synthetic target lists.
 *   aod_harness aodfile < scenarios.txt: aodfileread. */
static double now_ms(void){
  struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}

static unsigned bits(float v){ unsigned u; memcpy(&u, &v, 4); return u; }

static void rd(void *p, size_t sz, size_t n, FILE *f){
  if (fread(p, sz, n, f) != n){ fprintf(stderr, "short read\n"); exit(2); }
}
static void wr(FILE *f, const void *p, size_t sz, size_t n){ fwrite(p, sz, n, f); }

/* kind: 0 float, 1 short, 2 small, 3 ushort */
static brick_t *mkbrick(int nx, int ny, double res, int nb, int nodata, int kind){
  brick_t *b = calloc(1, sizeof(brick_t));
  b->nx = nx; b->ny = ny; b->nb = nb; b->nc = nx * ny; b->geotran[_GT_RES_] = res;
  for (int k = 0; k < 32; k++) b->nodata[k] = nodata;
  b->domain = calloc(nb > 0 ? nb : 1, 16);
  size_t n = (size_t)nx * ny;
  if (kind == 0){ b->vfloat = calloc(nb, sizeof(float*)); for (int k = 0; k < nb; k++) b->vfloat[k] = calloc(n, sizeof(float)); }
  if (kind == 1){ b->vshort = calloc(nb, sizeof(short*)); for (int k = 0; k < nb; k++) b->vshort[k] = calloc(n, sizeof(short)); }
  if (kind == 2){ b->vsmall = calloc(nb, sizeof(small*)); for (int k = 0; k < nb; k++) b->vsmall[k] = calloc(n, 1); }
  if (kind == 3){ b->vushort = calloc(nb, sizeof(ushort*)); for (int k = 0; k < nb; k++) b->vushort[k] = calloc(n, sizeof(ushort)); }
  return b;
}

static void dump_objects(FILE *f, darkobj_t *d, int num, int nb){
  wr(f, &num, 4, 1);
  for (int o = 0; o < num; o++){
    int v[12] = {d[o].k, d[o].valid, d[o].imin, d[o].imax, d[o].jmin, d[o].jmax, d[o].i, d[o].j, d[o].e, d[o].f, d[o].g, d[o].r};
    float x[7] = {d[o].ms, d[o].mv, d[o].razi, d[o].cosi, d[o].sky, d[o].z, d[o].Hr};
    wr(f, v, 4, 12); wr(f, x, 4, 7);
    wr(f, d[o].ttoa, 4, nb); wr(f, d[o].etoa, 4, nb);
  }
}

static void dump_libmatch(FILE *f, atc_t *atc, double res, bool multi, darkobj_t *d, int num, speclib_t *lib, int type, int nb){
  int nvalid = 0;
  for (int o = 0; o < num; o++) if (d[o].valid) nvalid++;
  wr(f, &nvalid, 4, 1);
  for (int o = 0; o < num; o++){
    if (!d[o].valid) continue;
    float **aodest = NULL;
    alloc_2D((void***)&aodest, lib->n, nb, sizeof(float));
    if (aod_lib_to_target(atc, res, multi, d[o], lib, type, aodest) == FAILURE){ fprintf(stderr, "lib_to_target failed\n"); exit(3); }
    for (int w = 0; w < lib->n; w++) wr(f, aodest[w], 4, nb);
    free_2D((void**)aodest, lib->n);
  }
}

static void dump_fit(FILE *f, darkobj_t *d, int num, int k, int nb){
  wr(f, &k, 4, 1);
  for (int o = 0; o < num; o++){
    int v[3] = {d[o].valid, d[o].lib_id, d[o].ang_fit};
    wr(f, v, 4, 3); wr(f, &d[o].rsq, 4, 1); wr(f, d[o].coef, 4, 3);
    wr(f, d[o].aod, 4, nb); wr(f, d[o].est, 4, nb);
  }
}

typedef struct {
  int nb, nx, ny, gnx, gny, gn, multi;
  double res;
  meta_t meta;
  atc_t atc;
  brick_t *TOA, *QAI;
  top_t top;
} sc_t;

static int load_scene(const char *fin, sc_t *sc){
  FILE *f = fopen(fin, "rb");
  if (!f){ perror("input"); return 2; }
  int nb, nx, ny, gnx, gny, nodata, multi;
  double res, gres;
  rd(&nb, 4, 1, f); rd(&nx, 4, 1, f); rd(&ny, 4, 1, f); rd(&res, 8, 1, f);
  rd(&gnx, 4, 1, f); rd(&gny, 4, 1, f); rd(&gres, 8, 1, f); rd(&nodata, 4, 1, f); rd(&multi, 4, 1, f);
  int gn = gnx * gny, npix = nx * ny;
  char (*domain)[16] = calloc(nb, 16);
  rd(domain, 16, nb, f);
  meta_t meta; meta.cal = calloc(nb, sizeof(cal_t));
  for (int b = 0; b < nb; b++){ int r; rd(&r, 4, 1, f); meta.cal[b].rsr_band = r; }
  atc_t atc; memset(&atc, 0, sizeof(atc));
  atc.aod_bands = calloc(nb, sizeof(bool));
  for (int b = 0; b < nb; b++){ unsigned char u; rd(&u, 1, 1, f); atc.aod_bands[b] = u; }
  atc.wvl = calloc(nb, 4); atc.lwvl = calloc(nb, 4); atc.E0 = calloc(nb, 4);
  rd(atc.wvl, 4, nb, f); rd(atc.lwvl, 4, nb, f); rd(atc.E0, 4, nb, f);
  rd(&atc.dem.avg, 4, 1, f); rd(&atc.dem.min, 4, 1, f); rd(&atc.dem.max, 4, 1, f); rd(&atc.dem.step, 4, 1, f);
  rd(&atc.tthg.sob, 4, 1, f);
  atc.xy_sun = mkbrick(gnx, gny, gres, 8, nodata, 0);
  atc.xy_view = mkbrick(gnx, gny, gres, 8, nodata, 0);
  atc.xy_aod = mkbrick(gnx, gny, gres, nb, nodata, 0);
  atc.xy_mod = mkbrick(gnx, gny, gres, nb, nodata, 0);
  atc.xy_Tso = mkbrick(gnx, gny, gres, nb, nodata, 0);
  atc.xy_Tvo = mkbrick(gnx, gny, gres, nb, nodata, 0);
  atc.xy_Tg = mkbrick(gnx, gny, gres, nb, nodata, 0);
  atc.xy_brdf = mkbrick(gnx, gny, gres, nb, nodata, 0);
  atc.xy_Pr = mkbrick(gnx, gny, gres, 1, nodata, 0);
  atc.xy_Pa = mkbrick(gnx, gny, gres, 1, nodata, 0);
  atc.xy_Hr = mkbrick(gnx, gny, gres, 1, nodata, 0);
  atc.xy_Ha = mkbrick(gnx, gny, gres, 1, nodata, 0);
  atc.xy_fresnel = mkbrick(gnx, gny, gres, 1, nodata, 0);
  atc.xy_dem = mkbrick(gnx, gny, gres, 1, nodata, 0);
  atc.xy_interp = mkbrick(gnx, gny, gres, 1, nodata, 0);
  atc.aod = calloc(nb, 4);
  for (int b = 0; b < nb; b++) memcpy(atc.xy_aod->domain[b], domain[b], 16);
  rd(atc.xy_sun->vfloat[cZEN], 4, gn, f); rd(atc.xy_sun->vfloat[AZI], 4, gn, f);
  rd(atc.xy_view->vfloat[cZEN], 4, gn, f); rd(atc.xy_view->vfloat[AZI], 4, gn, f);
  for (int b = 0; b < nb; b++) rd(atc.xy_Tso->vfloat[b], 4, gn, f);
  for (int b = 0; b < nb; b++) rd(atc.xy_Tvo->vfloat[b], 4, gn, f);
  for (int b = 0; b < nb; b++) rd(atc.xy_Tg->vfloat[b], 4, gn, f);
  for (int b = 0; b < nb; b++) rd(atc.xy_mod->vfloat[b], 4, gn, f);
  for (int b = 0; b < nb; b++) rd(atc.xy_brdf->vfloat[b], 4, gn, f);
  rd(atc.xy_Pr->vfloat[0], 4, gn, f); rd(atc.xy_Pa->vfloat[0], 4, gn, f); rd(atc.xy_Hr->vfloat[0], 4, gn, f);
  rd(atc.xy_fresnel->vfloat[0], 4, gn, f); rd(atc.xy_dem->vfloat[0], 4, gn, f);
  brick_t *TOA = mkbrick(nx, ny, res, nb, -9999, 1);
  brick_t *QAI = mkbrick(nx, ny, res, 1, -9999, 1);
  top_t top; memset(&top, 0, sizeof(top));
  top.dem = mkbrick(nx, ny, res, 1, -9999, 2);
  top.exp = mkbrick(nx, ny, res, 1, -9999, 3);
  top.ill = mkbrick(nx, ny, res, 1, -9999, 1);
  top.sky = mkbrick(nx, ny, res, 1, -9999, 3);
  for (int b = 0; b < nb; b++){ rd(TOA->vshort[b], 2, npix, f); memcpy(TOA->domain[b], domain[b], 16); }
  rd(QAI->vshort[0], 2, npix, f);
  rd(top.dem->vsmall[0], 1, npix, f); rd(top.exp->vushort[0], 2, npix, f);
  rd(top.ill->vshort[0], 2, npix, f); rd(top.sky->vushort[0], 2, npix, f);
  fclose(f);

  sc->nb = nb; sc->nx = nx; sc->ny = ny; sc->gnx = gnx; sc->gny = gny; sc->gn = gn; sc->multi = multi; sc->res = res;
  sc->meta = meta; sc->atc = atc; sc->TOA = TOA; sc->QAI = QAI; sc->top = top;
  return 0;
}

static int scene(const char *fin, const char *fout){
  sc_t sc;
  if (load_scene(fin, &sc)) return 2;
  int nb = sc.nb, gn = sc.gn, multi = sc.multi;
  double res = sc.res;
  meta_t meta = sc.meta; atc_t atc = sc.atc; brick_t *TOA = sc.TOA, *QAI = sc.QAI; top_t top = sc.top;

  int blue = find_domain(atc.xy_aod, "BLUE"), green = find_domain(atc.xy_aod, "GREEN");
  int red = find_domain(atc.xy_aod, "RED"), sw2 = find_domain(atc.xy_aod, "SWIR2");
  FILE *f = fopen(fout, "wb");
  double t0 = now_ms();
  dark_t dark; memset(&dark, 0, sizeof(dark));
  dark.kwat = extract_dark_target(&atc, TOA, QAI, &top, _AOD_WAT_, &dark.wat);
  dark.kshd = extract_dark_target(&atc, TOA, QAI, &top, _AOD_SHD_, &dark.shd);
  dark.kveg = extract_dark_target(&atc, TOA, QAI, &top, _AOD_VEG_, &dark.veg);
  double t1 = now_ms();
  if (dark.kwat < 0 || dark.kveg < 0){ fprintf(stderr, "extract_dark_target failed\n"); return 3; }
  dump_objects(f, dark.wat, dark.kwat, nb);
  dump_objects(f, dark.veg, dark.kveg, nb);

  speclib_t *wlib = water_lib(nb, &meta);
  dump_libmatch(f, &atc, res, multi, dark.wat, dark.kwat, wlib, _AOD_WAT_, nb);
  speclib_t *vlib = veg_lib(nb, blue, green, red);
  {
    /* what aod_from_target sets before the vegetation estimation */
    bool *save = calloc(nb, sizeof(bool));
    memcpy(save, atc.aod_bands, nb);
    for (int b = 0; b < nb; b++) atc.aod_bands[b] = false;
    atc.aod_bands[blue] = atc.aod_bands[green] = atc.aod_bands[red] = atc.aod_bands[sw2] = true;
    dump_libmatch(f, &atc, res, multi, dark.veg, dark.kveg, vlib, _AOD_VEG_, nb);
    memcpy(atc.aod_bands, save, nb);
    free(save);
  }

  par_ll_t pl2; pl2.d_aod = "NULL"; pl2.domulti = multi; pl2.doaod = true; pl2.d_log = "NULL";
  double t2 = now_ms();
  dark.nwat = aod_from_target(&pl2, &meta, &atc, res, dark.wat, dark.kwat, _AOD_WAT_);
  dark.nshd = aod_from_target(&pl2, &meta, &atc, res, dark.shd, dark.kshd, _AOD_SHD_);
  dark.nveg = aod_from_target(&pl2, &meta, &atc, res, dark.veg, dark.kveg, _AOD_VEG_);
  double t3 = now_ms();
  if (dark.nwat < 0 || dark.nveg < 0 || dark.nshd < 0){ fprintf(stderr, "aod_from_target failed\n"); return 3; }
  dump_fit(f, dark.wat, dark.kwat, dark.nwat, nb);
  dump_fit(f, dark.veg, dark.kveg, dark.nveg, nb);

  aod_elev_dependency(&atc, &dark, green);
  wr(f, &atc.Hp, 4, 1); wr(f, &atc.Ha, 4, 1);
  wr(f, atc.xy_Ha->vfloat[0], 4, gn);
  for (int o = 0; o < dark.kwat; o++) wr(f, &dark.wat[o].Ha, 4, 1);
  for (int o = 0; o < dark.kveg; o++) wr(f, &dark.veg[o].Ha, 4, 1);
  fclose(f);
  printf("%.3f %.3f\n", t1 - t0, t3 - t2);
  return 0;
}

/* compile_aod of FORCE as it is, on the scene of the `scene` mode (the sun zenith of the view brick, which FORCE fills
 * and the dark target functions do not read except for the swath test of the map, is 30 where there is a view and the
 * nodata value where there is none). */
static int compile(const char *fin, const char *fout, int doaod, const char *dir, double lon, double lat, int doy){
  sc_t sc;
  if (load_scene(fin, &sc)) return 2;
  int nb = sc.nb, gn = sc.gn;
  atc_t *atc = &sc.atc;
  for (int g = 0; g < gn; g++){
    float cz = atc->xy_view->vfloat[cZEN][g];
    atc->xy_view->vfloat[ZEN][g] = (cz < 0) ? cz : 30.0f;
  }
  atc->xy_aod->lon = lon; atc->xy_aod->lat = lat; atc->xy_aod->doy = doy;
  par_ll_t pl2; pl2.d_aod = (char*)dir; pl2.domulti = sc.multi; pl2.doaod = doaod; pl2.d_log = "NULL";
  double t0 = now_ms();
  int ret = compile_aod(&pl2, &sc.meta, atc, sc.TOA, sc.QAI, &sc.top);
  double t1 = now_ms();
  FILE *f = fopen(fout, "wb");
  int v[2] = {ret, atc->aodmap};
  wr(f, v, 4, 2);
  if (ret == SUCCESS){
    wr(f, atc->aod, 4, nb);
    wr(f, &atc->Hp, 4, 1); wr(f, &atc->Ha, 4, 1);
    for (int b = 0; b < nb; b++){ int u = atc->aod_bands[b]; wr(f, &u, 4, 1); }
    for (int b = 0; b < nb; b++) wr(f, atc->xy_aod->vfloat[b], 4, gn);
    wr(f, atc->xy_interp->vfloat[0], 4, gn);
    wr(f, atc->xy_Ha->vfloat[0], 4, gn);
  }
  fclose(f);
  printf("%.3f\n", t1 - t0);
  return 0;
}

/* GDALGridCreate as interpolate_aod_map calls it: in.bin: int32 nx, ny, np; float64 x[np], y[np], z[np] -> float64
 * grid[nx*ny] over (0, nx-1, 0, ny-1). */
static int idw(const char *fin, const char *fout){
  FILE *f = fopen(fin, "rb");
  if (!f){ perror("input"); return 2; }
  int nx, ny, np;
  rd(&nx, 4, 1, f); rd(&ny, 4, 1, f); rd(&np, 4, 1, f);
  double *x = calloc(np + 1, 8), *y = calloc(np + 1, 8), *z = calloc(np + 1, 8);
  rd(x, 8, np, f); rd(y, 8, np, f); rd(z, 8, np, f);
  fclose(f);
  GDALGridAlgorithm eAlgorithm = GGA_InverseDistanceToAPower;
  void *pOptions = NULL;
  ParseAlgorithmAndOptions(szAlgNameInvDist, &eAlgorithm, &pOptions);
  double *out = calloc((size_t)nx * ny, 8);
  double t0 = now_ms();
  GDALGridCreate(eAlgorithm, pOptions, np, x, y, z, 0, nx - 1, 0, ny - 1, nx, ny, GDT_Float64, (void*)out, NULL, NULL);
  double t1 = now_ms();
  CPLFree(pOptions);
  f = fopen(fout, "wb");
  wr(f, out, 8, (size_t)nx * ny);
  fclose(f);
  printf("%.3f\n", t1 - t0);
  return 0;
}

static int elev(void){
  int ncase;
  if (scanf("%d", &ncase) != 1) return 2;
  for (int c = 0; c < ncase; c++){
    float avg, mn, mx, step; int nt;
    if (scanf("%a %a %a %a %d", &avg, &mn, &mx, &step, &nt) != 5) return 2;
    atc_t atc; memset(&atc, 0, sizeof(atc));
    atc.dem.avg = avg; atc.dem.min = mn; atc.dem.max = mx; atc.dem.step = step;
    atc.xy_aod = mkbrick(2, 2, 1.0, 3, -9999, 0);
    atc.xy_view = mkbrick(2, 2, 1.0, 8, -9999, 0);
    atc.xy_Ha = mkbrick(2, 2, 1.0, 1, -9999, 0);
    atc.xy_dem = mkbrick(2, 2, 1.0, 1, -9999, 0);
    for (int g = 0; g < 4; g++){
      float v, d;
      if (scanf("%a %a", &v, &d) != 2) return 2;
      atc.xy_view->vfloat[0][g] = v;
      atc.xy_dem->vfloat[0][g] = d;
    }
    dark_t dark; memset(&dark, 0, sizeof(dark));
    dark.kwat = nt; dark.nwat = nt;
    dark.wat = calloc(nt > 0 ? nt : 1, sizeof(darkobj_t));
    for (int o = 0; o < nt; o++){
      float z, a;
      if (scanf("%a %a", &z, &a) != 2) return 2;
      dark.wat[o].valid = true; dark.wat[o].z = z;
      dark.wat[o].aod = calloc(3, 4); dark.wat[o].aod[1] = a;
    }
    aod_elev_dependency(&atc, &dark, 1);
    printf("%08x %08x", bits(atc.Hp), bits(atc.Ha));
    for (int g = 0; g < 4; g++) printf(" %08x", bits(atc.xy_Ha->vfloat[0][g]));
    for (int o = 0; o < nt; o++) printf(" %08x", bits(dark.wat[o].Ha));
    printf("\n");
  }
  return 0;
}

static int aodfile(void){
  int nb;
  if (scanf("%d", &nb) != 1) return 2;
  atc_t atc; memset(&atc, 0, sizeof(atc));
  atc.wvl = calloc(nb, 4); atc.lwvl = calloc(nb, 4);
  for (int b = 0; b < nb; b++) if (scanf("%a", &atc.wvl[b]) != 1) return 2;
  for (int b = 0; b < nb; b++) if (scanf("%a", &atc.lwvl[b]) != 1) return 2;
  atc.xy_aod = mkbrick(4, 4, 1.0, nb, -9999, 0);
  char dir[NPOW_10];
  double lon, lat; int doy;
  while (scanf("%1023s %lf %lf %d", dir, &lon, &lat, &doy) == 4){
    par_ll_t pl2; pl2.d_aod = dir; pl2.domulti = true; pl2.doaod = true; pl2.d_log = "NULL";
    atc.xy_aod->lon = lon; atc.xy_aod->lat = lat; atc.xy_aod->doy = doy;
    float *aod = aodfileread(&pl2, &atc);
    if (aod == NULL){ printf("NULL\n"); continue; }
    for (int b = 0; b < nb; b++) printf("%s%08x", b ? " " : "", bits(aod[b]));
    printf("\n");
    free(aod);
  }
  return 0;
}

static int image(const char *fin, const char *fout){
  FILE *f = fopen(fin, "rb");
  if (!f){ perror("input"); return 2; }
  int nx, ny;
  rd(&nx, 4, 1, f); rd(&ny, 4, 1, f);
  small *img = malloc((size_t)nx * ny);
  rd(img, 1, (size_t)nx * ny, f);
  fclose(f);
  ushort *dist = dist_transform_(img, nx, ny);
  int *lab = calloc((size_t)nx * ny, sizeof(int));
  int n = connectedcomponents_(img, lab, nx, ny);
  f = fopen(fout, "wb");
  wr(f, &n, 4, 1); wr(f, dist, 2, (size_t)nx * ny); wr(f, lab, 4, (size_t)nx * ny);
  fclose(f);
  return 0;
}

int main(int argc, char **argv){
  if (argc >= 4 && !strcmp(argv[1], "scene")) return scene(argv[2], argv[3]);
  if (argc >= 4 && !strcmp(argv[1], "image")) return image(argv[2], argv[3]);
  if (argc >= 8 && !strcmp(argv[1], "compile"))
    return compile(argv[2], argv[3], atoi(argv[4]), !strcmp(argv[5], "NULL") ? "NULL" : argv[5], atof(argv[6]), atof(argv[7]), argc > 8 ? atoi(argv[8]) : 1);
  if (argc >= 4 && !strcmp(argv[1], "idw")) return idw(argv[2], argv[3]);
  if (argc >= 2 && !strcmp(argv[1], "elev")) return elev();
  if (argc >= 2 && !strcmp(argv[1], "aodfile")) return aodfile();
  fprintf(stderr, "usage: %s scene in.bin out.bin | compile in.bin out.bin doaod dir lon lat doy | idw in.bin out.bin | elev < cases.txt | aodfile < scenarios.txt\n", argv[0]);
  return 2;
}
