/* The harness itself (Zeit), appended to atmo_harness.c by make_harness.sh after FORCE's functions.
 *
 *   atmo_harness scene in.bin out.bin
 *     in.bin (little endian): int32 mission (0 Landsat, 1 Sentinel-2), nx, ny; float64 res, ulx, uly (the image: pixels,
 *       metres, map coordinates of its upper left corner);
 *       int32 nb; per band: char domain[16], float32 wavelength, int32 rsr_band;
 *       int32 year, doy, hh, mm, ss, tz (the date of the image);
 *       float64 lon0, x0, kgeo (the warp: lon = lon0 + (x - x0) * kgeo, lat = y * kgeo);
 *       int32 domulti, doenv, aodmap; float32 dem.avg, min, max, step; int32 dem.cnum; float32 wvp, Hp, Ha;
 *       float32 mod[nb], aod[nb];
 *       int32 s2nx, s2ny; float32 s2nodata; if mission == 1: float32 vzen[s2ny*s2nx], vazi[s2ny*s2nx];
 *       int32 gn (the coarse cells the grid must have); float32 xy_aod[nb][gn], xy_interp[gn];
 *       int16 qai[nx*ny]; uint8 top_dem[nx*ny];
 *       int32 nsun; { float64 lat, lon; float32 zen, azi } [nsun] (the sun, see shims.h);
 *       int32 npts; if npts > 0: float64 wres, wfull_res; int32 wnf, wne; float32 wnodata; float32 coarse[wne*wnf];
 *         { int32 j, i } [npts]  (the interpolation weights);
 *       int32 naoi; if naoi: uint8 aoi[nx*ny] (apply_aoi; with naoi == 0, apply_aoi(QAI, NULL)).
 *     out.bin: see dump_* below (the same layout as zeit's binding writes): after allocate_atc, after sun_target_view, after
 *       atmo_angledep, after atmo_elevdep, the weights, apply_aoi; float64 times (ms) of the three stages. */
static double now_ms(void){
  struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}

static void rd(void *p, size_t sz, size_t n, FILE *f){
  if (fread(p, sz, n, f) != n){ fprintf(stderr, "short read\n"); exit(2); }
}
static void wi(FILE *f, int v){ fwrite(&v, 4, 1, f); }
static void wf(FILE *f, float v){ fwrite(&v, 4, 1, f); }
static void wd(FILE *f, double v){ fwrite(&v, 8, 1, f); }

/* one brick: its band count, size, resolution, no-data and domain of every band */
static void dump_brick_meta(FILE *o, brick_t *b){
  wi(o, b->nb); wi(o, b->nx); wi(o, b->ny); wd(o, b->geotran[_GT_RES_]);
  for (int k = 0; k < b->nb; k++) wi(o, b->nodata[k]);
  for (int k = 0; k < b->nb; k++) fwrite(b->domain[k], 1, 16, o);
}
/* every cell of one band, as float (the int and small bricks hold whole numbers) */
static void dump_band(FILE *o, brick_t *b, int k){
  for (int p = 0; p < b->nc; p++) wf(o, get_brick(b, k, p));
}
static void dump_all_bands(FILE *o, brick_t *b){
  for (int k = 0; k < b->nb; k++) dump_band(o, b, k);
}
/* z-major: for each elevation bin, every band */
static void dump_xyz(FILE *o, brick_t **xyz){
  for (int z = 0; z < _BYTE_LEN_; z++) dump_all_bands(o, xyz[z]);
}

static void dump_scalars(FILE *o, atc_t *atc){
  wf(o, atc->cosszen[0]); wf(o, atc->cosszen[1]); wf(o, atc->cosvzen[0]); wf(o, atc->cosvzen[1]);
}
static void dump_view(FILE *o, atc_t *atc){
  wf(o, atc->view.a); wf(o, atc->view.b); wf(o, atc->view.c); wf(o, atc->view.ab);
  wf(o, atc->view.geo_angle_nadir); wf(o, atc->view.H); wf(o, atc->view.H2);
}

static void dump_after_alloc(FILE *o, atc_t *atc, int nb){
  wi(o, atc->nx); wi(o, atc->ny); wi(o, atc->nc); wi(o, atc->res);
  wf(o, atc->nodata);
  dump_scalars(o, atc);
  wf(o, atc->wvp); wf(o, atc->cc); wf(o, atc->Hr); wf(o, atc->Ha); wf(o, atc->Hp);
  wf(o, atc->km); wf(o, atc->Fr); wf(o, atc->Fa);
  wf(o, atc->dem.avg); wf(o, atc->dem.min); wf(o, atc->dem.max); wf(o, atc->dem.step); wi(o, atc->dem.cnum);
  dump_view(o, atc);
  wf(o, atc->tthg.g1); wf(o, atc->tthg.g2); wf(o, atc->tthg.alpha);
  for (int k = 0; k < 6; k++) wf(o, atc->tthg.hg[k]);
  wf(o, atc->tthg.sob);
  wi(o, atc->aodmap);
  for (int b = 0; b < nb; b++){
    wf(o, atc->wvl[b]); wf(o, atc->lwvl[b]); wf(o, atc->lwvl2[b]); wf(o, atc->E0[b]);
    wf(o, atc->od[b]); wf(o, atc->mod[b]); wf(o, atc->aod[b]); wi(o, atc->aod_bands[b]);
  }
  brick_t *xy[18] = { atc->xy_mod, atc->xy_aod, atc->xy_Pr, atc->xy_Pa, atc->xy_Hr, atc->xy_Ha, atc->xy_Tvw, atc->xy_Tsw,
    atc->xy_Tvo, atc->xy_Tso, atc->xy_Tg, atc->xy_brdf, atc->xy_fresnel, atc->xy_dem, atc->xy_interp, atc->xy_view,
    atc->xy_sun, atc->xy_psi };
  for (int k = 0; k < 18; k++) dump_brick_meta(o, xy[k]);
  brick_t **xyz[16] = { atc->xyz_od, atc->xyz_mod, atc->xyz_aod, atc->xyz_Hr, atc->xyz_Ha, atc->xyz_rho_p, atc->xyz_Ts,
    atc->xyz_tsd, atc->xyz_tss, atc->xyz_Tv, atc->xyz_tvd, atc->xyz_tvs, atc->xyz_T, atc->xyz_s, atc->xyz_F, atc->xyz_Tg };
  for (int k = 0; k < 16; k++){ dump_brick_meta(o, xyz[k][0]); dump_brick_meta(o, xyz[k][_BYTE_LEN_ - 1]); }
}

static int scene(const char *fin, const char *fout){
  FILE *f = fopen(fin, "rb");
  if (!f){ perror("input"); return 2; }
  int mission, nx, ny, nb;
  double res, ulx, uly;
  rd(&mission, 4, 1, f); rd(&nx, 4, 1, f); rd(&ny, 4, 1, f);
  rd(&res, 8, 1, f); rd(&ulx, 8, 1, f); rd(&uly, 8, 1, f);
  rd(&nb, 4, 1, f);
  int npix = nx * ny;

  brick_t *DN = calloc(1, sizeof(brick_t));
  alloc_brick_meta(DN, nb);
  DN->geotran[_GT_ULX_] = ulx; DN->geotran[_GT_ULY_] = uly;
  DN->geotran[_GT_XRES_] = res; DN->geotran[_GT_YRES_] = -res;
  DN->nx = nx; DN->ny = ny; DN->nc = npix; DN->width = nx * res; DN->height = ny * res;
  meta_t meta; memset(&meta, 0, sizeof(meta));
  meta.cal = calloc(nb, sizeof(cal_t));
  for (int b = 0; b < nb; b++){
    rd(DN->domain[b], 1, 16, f); rd(&DN->wavelength[b], 4, 1, f); rd(&meta.cal[b].rsr_band, 4, 1, f);
    DN->nodata[b] = -9999;
  }
  int year, doy, hh, mm, ss, tz;
  rd(&year, 4, 1, f); rd(&doy, 4, 1, f); rd(&hh, 4, 1, f); rd(&mm, 4, 1, f); rd(&ss, 4, 1, f); rd(&tz, 4, 1, f);
  for (int b = 0; b < nb; b++){
    memset(&DN->date[b], 0, sizeof(date_t));
    DN->date[b].year = year; DN->date[b].doy = doy; DN->date[b].hh = hh; DN->date[b].mm = mm; DN->date[b].ss = ss; DN->date[b].tz = tz;
  }
  rd(&g_lon0, 8, 1, f); rd(&g_x0, 8, 1, f); rd(&g_kgeo, 8, 1, f);

  par_ll_t pl2;
  int domulti, doenv, aodmap, cnum;
  float dem4[4], wvp, hp, ha;
  rd(&domulti, 4, 1, f); rd(&doenv, 4, 1, f); rd(&aodmap, 4, 1, f);
  rd(dem4, 4, 4, f); rd(&cnum, 4, 1, f); rd(&wvp, 4, 1, f); rd(&hp, 4, 1, f); rd(&ha, 4, 1, f);
  pl2.domulti = domulti; pl2.doenv = doenv;
  float *mod = calloc(nb, 4), *aod = calloc(nb, 4);
  rd(mod, 4, nb, f); rd(aod, 4, nb, f);

  int s2nx, s2ny; float s2nodata;
  rd(&s2nx, 4, 1, f); rd(&s2ny, 4, 1, f); rd(&s2nodata, 4, 1, f);
  meta.s2.nodata = s2nodata; meta.s2.nx = s2nx; meta.s2.ny = s2ny;
  if (mission == 1){
    meta.s2.vzen = calloc(s2ny, sizeof(float*)); meta.s2.vazi = calloc(s2ny, sizeof(float*));
    for (int e = 0; e < s2ny; e++){ meta.s2.vzen[e] = calloc(s2nx, 4); rd(meta.s2.vzen[e], 4, s2nx, f); }
    for (int e = 0; e < s2ny; e++){ meta.s2.vazi[e] = calloc(s2nx, 4); rd(meta.s2.vazi[e], 4, s2nx, f); }
  }

  int gn;
  rd(&gn, 4, 1, f);
  float *xy_aod_in = calloc((size_t)nb * gn, 4), *xy_interp_in = calloc(gn, 4);
  rd(xy_aod_in, 4, (size_t)nb * gn, f); rd(xy_interp_in, 4, gn, f);

  brick_t *QAI = copy_brick(DN, 1, _DT_SHORT_);
  rd(QAI->vshort[0], 2, npix, f);
  short *qai0 = malloc(npix * sizeof(short)); memcpy(qai0, QAI->vshort[0], npix * sizeof(short));
  top_t TOP; memset(&TOP, 0, sizeof(TOP));
  TOP.dem = copy_brick(DN, 1, _DT_SMALL_);
  rd(TOP.dem->vsmall[0], 1, npix, f);

  int nsun; rd(&nsun, 4, 1, f);
  g_nsun = nsun; g_sun = calloc(nsun > 0 ? nsun : 1, sizeof(sun_entry_t));
  for (int k = 0; k < nsun; k++){
    rd(&g_sun[k].lat, 8, 1, f); rd(&g_sun[k].lon, 8, 1, f); rd(&g_sun[k].zen, 4, 1, f); rd(&g_sun[k].azi, 4, 1, f);
  }

  int npts = 0, wnf = 0, wne = 0; double wres = 0, wfull = 0; float wnodata = 0;
  float *coarse = NULL; int *wji = NULL;
  rd(&npts, 4, 1, f);
  if (npts > 0){
    rd(&wres, 8, 1, f); rd(&wfull, 8, 1, f); rd(&wnf, 4, 1, f); rd(&wne, 4, 1, f); rd(&wnodata, 4, 1, f);
    coarse = calloc((size_t)wnf * wne, 4); rd(coarse, 4, (size_t)wnf * wne, f);
    wji = calloc(2 * npts, 4); rd(wji, 4, 2 * npts, f);
  }
  int naoi; rd(&naoi, 4, 1, f);
  brick_t *AOI = NULL;
  if (naoi){ AOI = copy_brick(DN, 1, _DT_SMALL_); rd(AOI->vsmall[0], 1, npix, f); }
  fclose(f);

  FILE *o = fopen(fout, "wb");
  if (!o){ perror("output"); return 2; }

  /* allocate_atc */
  atc_t *atc = allocate_atc(&pl2, &meta, DN);
  if (atc->nc != gn){ fprintf(stderr, "the grid has %d cells, the input %d\n", atc->nc, gn); return 2; }
  dump_after_alloc(o, atc, nb);

  /* what the stages before set */
  atc->dem.avg = dem4[0]; atc->dem.min = dem4[1]; atc->dem.max = dem4[2]; atc->dem.step = dem4[3]; atc->dem.cnum = cnum;
  atc->wvp = wvp; atc->Hp = hp; atc->Ha = ha; atc->aodmap = aodmap;
  for (int b = 0; b < nb; b++){ atc->mod[b] = mod[b]; atc->aod[b] = aod[b]; }
  for (int b = 0; b < nb; b++) for (int g = 0; g < gn; g++) atc->xy_aod->vfloat[b][g] = xy_aod_in[(size_t)b * gn + g];
  for (int g = 0; g < gn; g++) atc->xy_interp->vsmall[0][g] = (small)xy_interp_in[g];

  /* sun_target_view */
  double t0 = now_ms();
  int ret = sun_target_view(&pl2, &meta, mission == 0 ? LANDSAT : SENTINEL2, atc, QAI);
  double t_sun = now_ms() - t0;
  wi(o, ret);
  dump_all_bands(o, atc->xy_sun); dump_all_bands(o, atc->xy_view);
  dump_scalars(o, atc); dump_view(o, atc);
  fwrite(QAI->vshort[0], 2, npix, o);

  /* atmo_angledep */
  t0 = now_ms();
  ret = atmo_angledep(&pl2, &meta, atc, &TOP, QAI);
  double t_angle = now_ms() - t0;
  wi(o, ret);
  dump_all_bands(o, atc->xy_brdf); dump_all_bands(o, atc->xy_psi); dump_all_bands(o, atc->xy_Pr); dump_all_bands(o, atc->xy_Pa);
  dump_all_bands(o, atc->xy_Tsw); dump_all_bands(o, atc->xy_Tvw); dump_all_bands(o, atc->xy_Tso); dump_all_bands(o, atc->xy_Tvo);
  dump_all_bands(o, atc->xy_Tg); dump_all_bands(o, atc->xy_dem); dump_all_bands(o, atc->xy_Hr); dump_all_bands(o, atc->xy_mod);
  dump_all_bands(o, atc->xy_fresnel);

  /* atmo_elevdep */
  t0 = now_ms();
  ret = atmo_elevdep(&pl2, atc, QAI, &TOP);
  double t_elev = now_ms() - t0;
  wi(o, ret);
  dump_xyz(o, atc->xyz_od); dump_xyz(o, atc->xyz_mod); dump_xyz(o, atc->xyz_aod); dump_xyz(o, atc->xyz_rho_p);
  dump_xyz(o, atc->xyz_Ts); dump_xyz(o, atc->xyz_tsd); dump_xyz(o, atc->xyz_tss); dump_xyz(o, atc->xyz_Tv);
  dump_xyz(o, atc->xyz_tvd); dump_xyz(o, atc->xyz_tvs); dump_xyz(o, atc->xyz_T); dump_xyz(o, atc->xyz_s);
  dump_xyz(o, atc->xyz_F); dump_xyz(o, atc->xyz_Hr); dump_xyz(o, atc->xyz_Ha);
  fwrite(QAI->vshort[0], 2, npix, o);

  /* the interpolation weights and the interpolated values */
  for (int k = 0; k < npts; k++){
    iweights_t w = interpolation_weights(wji[2 * k], wji[2 * k + 1], wnf, wne, wres, wfull, coarse, wnodata);
    wi(o, w.gul); wi(o, w.gur); wi(o, w.gll); wi(o, w.glr);
    wf(o, w.wul); wf(o, w.wur); wf(o, w.wll); wf(o, w.wlr); wf(o, w.wtop); wf(o, w.wdown);
    wf(o, interpolate_coarse(w, coarse));
  }

  /* apply_aoi on the QAI as it came in */
  memcpy(QAI->vshort[0], qai0, npix * sizeof(short));
  ret = apply_aoi(QAI, AOI);
  wi(o, ret);
  fwrite(QAI->vshort[0], 2, npix, o);

  wd(o, t_sun); wd(o, t_angle); wd(o, t_elev);
  fclose(o);
  return 0;
}

int main(int argc, char **argv){
  if (argc == 4 && strcmp(argv[1], "scene") == 0) return scene(argv[2], argv[3]);
  fprintf(stderr, "usage: atmo_harness scene in.bin out.bin\n");
  return 2;
}
