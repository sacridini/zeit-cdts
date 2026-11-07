/* The harness itself (Zeit), appended to topo_harness.c by make_harness.sh after FORCE's functions.
 *
 *   topo_harness scene in.bin out.bin
 *     in.bin (little endian): int32 nx, ny; float64 res; int32 gnx, gny; float64 gres; int32 dem_nodata, flat, do_c;
 *       coarse grid (gn = gnx*gny): float32 sun_zen[gn], sun_azi[gn], sun_czen[gn], sun_szen[gn];
 *       pixels (npix = nx*ny): float32 dem[npix] (the DEM on the grid of the scene, as the warp would give it; ignored if
 *       flat: FORCE's FILE_DEM = NULL); int16 qai[npix];
 *       if do_c: int32 nb; char domain[nb][16]; int16 toa[nb][npix]; and for the band of the domain SWIR2 (the others are
 *       zero) float32 rho_p[256][gn], tss[256][gn], tsd[256][gn].
 *     out.bin: int32 ret of compile_topography; float32 dem.avg, min, max, step; int32 dem.cnum; int16 qai[npix] (as
 *       compile_topography leaves it); if ret == 0: uint8 dem[npix] (binned), uint16 slope[npix], aspect[npix], int16
 *       ill[npix], uint16 sky[npix]; if do_c: int32 (0: the brick, 1: NULL), uint16 cfactor[npix]; int32 xy_dem[gn]
 *       (average_elevation_cell of every coarse cell); float64 time of compile_topography, of cfactor_topography (ms). */
static double now_ms(void){
  struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}

static void rd(void *p, size_t sz, size_t n, FILE *f){
  if (fread(p, sz, n, f) != n){ fprintf(stderr, "short read\n"); exit(2); }
}
static void wr(FILE *f, const void *p, size_t sz, size_t n){ fwrite(p, sz, n, f); }

static brick_t *grid_brick(int nx, int ny, double res, int nb, int dt){
  brick_t proto; memset(&proto, 0, sizeof(proto));
  proto.nx = nx; proto.ny = ny; proto.nc = nx * ny; proto.geotran[_GT_RES_] = res;
  return copy_brick(&proto, nb, dt);
}

static int scene(const char *fin, const char *fout){
  FILE *f = fopen(fin, "rb");
  if (!f){ perror("input"); return 2; }
  int nx, ny, gnx, gny, dem_nodata, flat, do_c;
  double res, gres;
  rd(&nx, 4, 1, f); rd(&ny, 4, 1, f); rd(&res, 8, 1, f);
  rd(&gnx, 4, 1, f); rd(&gny, 4, 1, f); rd(&gres, 8, 1, f);
  rd(&dem_nodata, 4, 1, f); rd(&flat, 4, 1, f); rd(&do_c, 4, 1, f);
  int gn = gnx * gny, npix = nx * ny;

  atc_t atc; memset(&atc, 0, sizeof(atc));
  atc.xy_sun = grid_brick(gnx, gny, gres, 8, _DT_FLOAT_);
  rd(atc.xy_sun->vfloat[ZEN], 4, gn, f); rd(atc.xy_sun->vfloat[AZI], 4, gn, f);
  rd(atc.xy_sun->vfloat[cZEN], 4, gn, f); rd(atc.xy_sun->vfloat[sZEN], 4, gn, f);
  atc.xy_dem = grid_brick(gnx, gny, gres, 1, _DT_INT_);

  g_dem_in = calloc(npix, sizeof(float));
  rd(g_dem_in, 4, npix, f);
  brick_t *QAI = grid_brick(nx, ny, res, 1, _DT_SHORT_);
  rd(QAI->vshort[0], 2, npix, f);

  brick_t *TOA = NULL;
  int nb = 0, sw2 = -1;
  if (do_c){
    rd(&nb, 4, 1, f);
    TOA = grid_brick(nx, ny, res, nb, _DT_SHORT_);
    rd(TOA->domain, 16, nb, f);
    for (int b = 0; b < nb; b++) rd(TOA->vshort[b], 2, npix, f);
    sw2 = find_domain(TOA, "SWIR2");
    atc.xyz_rho_p = calloc(_BYTE_LEN_, sizeof(brick_t*));
    atc.xyz_tss = calloc(_BYTE_LEN_, sizeof(brick_t*));
    atc.xyz_tsd = calloc(_BYTE_LEN_, sizeof(brick_t*));
    for (int z = 0; z < _BYTE_LEN_; z++){
      atc.xyz_rho_p[z] = grid_brick(gnx, gny, gres, nb, _DT_FLOAT_);
      atc.xyz_tss[z] = grid_brick(gnx, gny, gres, nb, _DT_FLOAT_);
      atc.xyz_tsd[z] = grid_brick(gnx, gny, gres, nb, _DT_FLOAT_);
    }
    if (sw2 >= 0){
      for (int z = 0; z < _BYTE_LEN_; z++) rd(atc.xyz_rho_p[z]->vfloat[sw2], 4, gn, f);
      for (int z = 0; z < _BYTE_LEN_; z++) rd(atc.xyz_tss[z]->vfloat[sw2], 4, gn, f);
      for (int z = 0; z < _BYTE_LEN_; z++) rd(atc.xyz_tsd[z]->vfloat[sw2], 4, gn, f);
    }
  }
  fclose(f);

  par_ll_t pl2; memset(&pl2, 0, sizeof(pl2));
  pl2.fdem = flat ? "NULL" : "dem.tif";
  pl2.dem_nodata = dem_nodata; pl2.dem_resample = 1; pl2.nthread = 1;

  top_t *top = NULL;
  double t0 = now_ms();
  int ret = compile_topography(&pl2, &atc, &top, QAI);
  double t1 = now_ms();

  FILE *o = fopen(fout, "wb");
  wr(o, &ret, 4, 1);
  wr(o, &atc.dem.avg, 4, 1); wr(o, &atc.dem.min, 4, 1); wr(o, &atc.dem.max, 4, 1); wr(o, &atc.dem.step, 4, 1);
  wr(o, &atc.dem.cnum, 4, 1);
  wr(o, QAI->vshort[0], 2, npix);
  double t2 = t1, t3 = t1;
  if (ret == SUCCESS){
    wr(o, top->dem->vsmall[0], 1, npix);
    wr(o, top->exp->vushort[ZEN], 2, npix); wr(o, top->exp->vushort[AZI], 2, npix);
    wr(o, top->ill->vshort[0], 2, npix); wr(o, top->sky->vushort[0], 2, npix);
    if (do_c){
      t2 = now_ms();
      brick_t *CF = cfactor_topography(&atc, TOA, QAI, top->dem, top->exp, top->ill);
      t3 = now_ms();
      int cnull = (CF == NULL);
      wr(o, &cnull, 4, 1);
      if (CF) wr(o, CF->vushort[0], 2, npix); else { ushort *z = calloc(npix, 2); wr(o, z, 2, npix); }
    }
    for (int g = 0; g < gn; g++) average_elevation_cell(g, atc.xy_dem, top->dem, QAI);
    wr(o, atc.xy_dem->vint[0], 4, gn);
  }
  double tt[2] = {t1 - t0, t3 - t2};
  wr(o, tt, 8, 2);
  fclose(o);
  return 0;
}

int main(int argc, char **argv){
  if (argc == 4 && strcmp(argv[1], "scene") == 0) return scene(argv[2], argv[3]);
  fprintf(stderr, "usage: topo_harness scene in.bin out.bin\n");
  return 1;
}
