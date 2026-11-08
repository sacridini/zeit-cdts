/* The harness of the Level 2 comparison (Zeit), compiled against FORCE 3.11.00's own headers and linked with FORCE's own
 * object files (the ones `make` built for `force-l2ps`: atmo-ll, aod-ll, gas-ll, topo-ll, atc-ll, sunview-ll,
 * radtran-ll, brick_base-cl, quality-cl, ...), so that `radiometric_correction` runs as FORCE's L2PS runs it, with the
 * same functions in the same order (see make_harness.sh). Copyright (C) 2013-2022 David Frantz, GPL-3.0-or-later, for
 * what it links; this file is Zeit's, same license.
 *
 * Replaced (the same on both sides, Zeit's `radtran::sunpos` being clean-room): the sun position (`sunpos`: a table of the
 * values Zeit's sunpos gives for every coarse cell, looked up by the latitude and longitude), the warp of map coordinates
 * to longitude and latitude (`warp_any_to_geo`: an affine map), the warp of the DEM file (`warp_from_disc_to_known_brick`:
 * the DEM the harness holds) and `brdf_factor` (brdf-ll.c is not ported: a factor that depends on the view zenith and the
 * band, computed in float, that Zeit's side reproduces).
 *
 *   l2_harness scene in.bin out.bin
 *     in.bin (little endian): int32 mission (0 Landsat, 1 Sentinel-2), nx, ny; float64 res, ulx, uly;
 *       int32 nb; per band: char domain[16], float32 wavelength, int32 rsr_band;
 *       int32 year, doy, hh, mm, ss, tz;
 *       float64 lon0, x0, kgeo (the warp: lon = lon0 + (x - x0) * kgeo, lat = y * kgeo);
 *       int32 domulti, doenv, doatmo, dotopo, dobrdf, erase_cloud, doaod, odst, ovzn, ohot, oaod, owvp;
 *       float32 wvp (the constant of the parameter file);
 *       int32 dem_nodata, hasdem; if hasdem: float32 dem[nx*ny];
 *       int32 s2nx, s2ny; float32 s2nodata; if mission == 1: float32 vzen[s2ny*s2nx], vazi[s2ny*s2nx];
 *       int32 gn (the coarse cells the grid must have);
 *       int16 qai[nx*ny]; int16 toa[nb][nx*ny];
 *       int32 nsun; { float64 lat, lon; float32 zen, azi } [nsun];
 *       int32 naoi; if naoi: uint8 aoi[nx*ny].
 *     out.bin: int32 status (0 ok, 1 radiometric_correction failed); float64 ms; the dump of the atc (every brick: int32
 *       bands, then the bands; the families of 256 bricks: int32 bands, then z-major), then the products (see dump_* below;
 *       zeit's binding writes the same layout); then int32 nb and nb maps of the gaseous transmittance (int16, from the
 *       final QAI) for Sentinel-2 with doatmo and owvp, else int32 0. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#include "atmo-ll.h"
#include "../cross-level/sun-cl.h"
#include "../cross-level/warp-cl.h"
#include "../cross-level/date-cl.h"

/* ---------------------------------------------------------------------------------------------------------------- */
/* what is replaced                                                                                                  */
/* ---------------------------------------------------------------------------------------------------------------- */

typedef struct { double lat, lon; float zen, azi; } sun_entry_t;
static sun_entry_t *g_sun = NULL;
static int g_nsun = 0;
void sunpos(float latitude, float longitude, date_t date, float *zen, float *azi){
  (void)date;
  for (int k = 0; k < g_nsun; k++){
    if ((float)g_sun[k].lat == latitude && (float)g_sun[k].lon == longitude){ *zen = g_sun[k].zen; *azi = g_sun[k].azi; return; }
  }
  fprintf(stderr, "sunpos: no entry for lat %.9g lon %.9g\n", latitude, longitude);
  exit(3);
}

static double g_lon0 = 0, g_x0 = 0, g_kgeo = 1;
int warp_any_to_geo(double srs_x, double srs_y, double *dst_x, double *dst_y, char *src_wkt){
  (void)src_wkt;
  *dst_x = g_lon0 + (srs_x - g_x0) * g_kgeo;
  *dst_y = srs_y * g_kgeo;
  return SUCCESS;
}

static float *g_dem_in = NULL;
int warp_from_disc_to_known_brick(int rsm, int threads, const char *fname, brick_t *dst, int src_b, int dst_b, int src_nodata){
  (void)rsm; (void)threads; (void)fname; (void)src_b; (void)src_nodata;
  memcpy(get_band_float(dst, dst_b), g_dem_in, (size_t)get_brick_ncells(dst) * sizeof(float));
  return SUCCESS;
}

/* brdf_factor of brdf-ll.c is not ported: a factor of the view zenith (radians, float) and of the band, in float */
int brdf_factor(brick_t *sun, brick_t *view, brick_t *cor, int g){
  (void)sun;
  float zen = get_brick(view, ZEN, g);
  for (int b = 0; b < get_brick_nbands(cor); b++) set_brick(cor, b, g, 1.0f + 0.5f * zen * (1.0f + 0.1f * (float)b));
  return SUCCESS;
}

/* ---------------------------------------------------------------------------------------------------------------- */

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

static void dump_all_bands(FILE *o, brick_t *b){
  int nc = get_brick_ncells(b);
  wi(o, get_brick_nbands(b));
  for (int k = 0; k < get_brick_nbands(b); k++) for (int p = 0; p < nc; p++) wf(o, get_brick(b, k, p));
}
static void dump_xyz(FILE *o, brick_t **xyz){
  int nc = get_brick_ncells(xyz[0]);
  wi(o, get_brick_nbands(xyz[0]));
  for (int z = 0; z < _BYTE_LEN_; z++)
    for (int k = 0; k < get_brick_nbands(xyz[z]); k++) for (int p = 0; p < nc; p++) wf(o, get_brick(xyz[z], k, p));
}

static void dump_atc(FILE *o, atc_t *atc, int nb){
  wf(o, atc->wvp); wf(o, atc->Hr); wf(o, atc->Ha); wf(o, atc->Hp); wi(o, atc->aodmap);
  wf(o, atc->dem.avg); wf(o, atc->dem.min); wf(o, atc->dem.max); wf(o, atc->dem.step); wi(o, atc->dem.cnum);
  wf(o, atc->cosszen[0]); wf(o, atc->cosszen[1]); wf(o, atc->cosvzen[0]); wf(o, atc->cosvzen[1]);
  for (int b = 0; b < nb; b++){ wf(o, atc->mod[b]); wf(o, atc->aod[b]); wf(o, atc->od[b]); wi(o, atc->aod_bands[b]); }
  brick_t *xy[18] = { atc->xy_mod, atc->xy_aod, atc->xy_Pr, atc->xy_Pa, atc->xy_Hr, atc->xy_Ha, atc->xy_Tvw, atc->xy_Tsw,
    atc->xy_Tvo, atc->xy_Tso, atc->xy_Tg, atc->xy_brdf, atc->xy_fresnel, atc->xy_dem, atc->xy_interp, atc->xy_view,
    atc->xy_sun, atc->xy_psi };
  for (int k = 0; k < 18; k++) dump_all_bands(o, xy[k]);
  brick_t **xyz[15] = { atc->xyz_od, atc->xyz_mod, atc->xyz_aod, atc->xyz_rho_p, atc->xyz_Ts, atc->xyz_tsd, atc->xyz_tss,
    atc->xyz_Tv, atc->xyz_tvd, atc->xyz_tvs, atc->xyz_T, atc->xyz_s, atc->xyz_F, atc->xyz_Hr, atc->xyz_Ha };
  for (int k = 0; k < 15; k++) dump_xyz(o, xyz[k]);
}

static void dump_product(FILE *o, brick_t *b){
  int nb = get_brick_nbands(b), nc = get_brick_ncells(b);
  char domain[NPOW_10];
  wi(o, nb);
  for (int k = 0; k < nb; k++){
    char d16[16] = {0};
    get_brick_domain(b, k, domain, NPOW_10);
    memcpy(d16, domain, strnlen(domain, 15));
    fwrite(d16, 1, 16, o);
    wf(o, get_brick_wavelength(b, k));
  }
  for (int k = 0; k < nb; k++) fwrite(get_band_short(b, k), 2, nc, o);
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

  /* the digital numbers brick: what allocate_atc reads (size, resolution, wavelength, domain) and the date; the strings
   * that copy_brick copies are set as the metadata parser sets them */
  brick_t *DN = allocate_brick(nb, npix, _DT_SHORT_);
  set_brick_res(DN, res); set_brick_ncols(DN, nx); set_brick_nrows(DN, ny);
  set_brick_ulx(DN, ulx); set_brick_uly(DN, uly);
  set_brick_proj(DN, "harness");
  set_brick_name(DN, "harness"); set_brick_product(DN, "DN"); set_brick_dirname(DN, "."); set_brick_filename(DN, "harness");
  set_brick_provdir(DN, "."); set_brick_par(DN, "harness");
  set_brick_nprovenance(DN, 1); set_brick_provenance(DN, 0, "harness");
  meta_t *meta = allocate_metadata();
  meta->cal = allocate_calibration(nb);
  for (int b = 0; b < nb; b++){
    char dom[16]; float wvl; int rsr;
    rd(dom, 1, 16, f); rd(&wvl, 4, 1, f); rd(&rsr, 4, 1, f);
    set_brick_domain(DN, b, dom); set_brick_wavelength(DN, b, wvl); set_brick_nodata(DN, b, -9999);
    meta->cal[b].rsr_band = rsr;
  }
  int year, doy, hh, mm, ss, tz;
  rd(&year, 4, 1, f); rd(&doy, 4, 1, f); rd(&hh, 4, 1, f); rd(&mm, 4, 1, f); rd(&ss, 4, 1, f); rd(&tz, 4, 1, f);
  date_t date; init_date(&date);
  set_date_year(&date, year); set_date_doy(&date, doy);
  date.hh = hh; date.mm = mm; date.ss = ss; date.tz = tz;
  for (int b = 0; b < nb; b++) set_brick_date(DN, b, date);
  rd(&g_lon0, 8, 1, f); rd(&g_x0, 8, 1, f); rd(&g_kgeo, 8, 1, f);

  int fl[12];
  rd(fl, 4, 12, f);
  float wvp_const; rd(&wvp_const, 4, 1, f);
  int dem_nodata, hasdem; rd(&dem_nodata, 4, 1, f); rd(&hasdem, 4, 1, f);
  if (hasdem){ g_dem_in = malloc((size_t)npix * 4); rd(g_dem_in, 4, npix, f); }

  par_ll_t *pl2 = allocate_param_lower();
  pl2->domulti = fl[0]; pl2->doenv = fl[1]; pl2->doatmo = fl[2]; pl2->dotopo = fl[3]; pl2->dobrdf = fl[4];
  pl2->erase_cloud = fl[5]; pl2->doaod = fl[6]; pl2->odst = fl[7]; pl2->ovzn = fl[8]; pl2->ohot = fl[9];
  pl2->oaod = fl[10]; pl2->owvp = fl[11]; pl2->oovv = 0;
  pl2->wvp = wvp_const; pl2->wvp_strict = 0;
  default_gdaloptions(_FMT_GTIFF_, &pl2->gdalopt);
  pl2->d_aod = strdup("NULL"); pl2->d_wvp = strdup("NULL"); pl2->d_log = strdup("NULL");
  pl2->fdem = strdup(hasdem ? "DEM" : "NULL");
  pl2->dem_nodata = dem_nodata; pl2->dem_resample = 1;
  pl2->doreproj = 0; pl2->dotile = 0; pl2->resample = 0; pl2->nthread = 1;
  snprintf(pl2->b_level1, NPOW_10, "harness");

  int s2nx, s2ny; float s2nodata;
  rd(&s2nx, 4, 1, f); rd(&s2ny, 4, 1, f); rd(&s2nodata, 4, 1, f);
  meta->s2.nodata = s2nodata; meta->s2.nx = s2nx; meta->s2.ny = s2ny;
  if (mission == 1){
    alloc_2D((void***)&meta->s2.vzen, s2ny, s2nx, sizeof(float));
    alloc_2D((void***)&meta->s2.vazi, s2ny, s2nx, sizeof(float));
    for (int e = 0; e < s2ny; e++) rd(meta->s2.vzen[e], 4, s2nx, f);
    for (int e = 0; e < s2ny; e++) rd(meta->s2.vazi[e], 4, s2nx, f);
  }
  int gn; rd(&gn, 4, 1, f);

  brick_t *QAI = copy_brick(DN, 1, _DT_SHORT_);
  rd(get_band_short(QAI, 0), 2, npix, f);
  brick_t *TOA = copy_brick(DN, nb, _DT_SHORT_);
  for (int b = 0; b < nb; b++) rd(get_band_short(TOA, b), 2, npix, f);

  int nsun; rd(&nsun, 4, 1, f);
  g_nsun = nsun; g_sun = calloc(nsun > 0 ? nsun : 1, sizeof(sun_entry_t));
  for (int k = 0; k < nsun; k++){
    rd(&g_sun[k].lat, 8, 1, f); rd(&g_sun[k].lon, 8, 1, f); rd(&g_sun[k].zen, 4, 1, f); rd(&g_sun[k].azi, 4, 1, f);
  }
  int naoi; rd(&naoi, 4, 1, f);
  brick_t *AOI = NULL;
  if (naoi){ AOI = copy_brick(DN, 1, _DT_SMALL_); rd(get_band_small(AOI, 0), 1, npix, f); }
  fclose(f);

  FILE *o = fopen(fout, "wb");
  if (!o){ perror("output"); return 2; }

  /* what the L2PS does before radiometric_correction */
  atc_t *atc = allocate_atc(pl2, meta, DN);
  if (atc == NULL){ fprintf(stderr, "allocate_atc failed\n"); return 2; }
  if (get_brick_ncells(atc->xy_sun) != gn){ fprintf(stderr, "the grid has %d cells, the input %d\n", get_brick_ncells(atc->xy_sun), gn); return 2; }
  if (sun_target_view(pl2, meta, mission == 0 ? LANDSAT : SENTINEL2, atc, QAI) != SUCCESS){ fprintf(stderr, "sun_target_view failed\n"); return 2; }
  top_t *TOP = NULL;
  if (compile_topography(pl2, atc, &TOP, QAI) != SUCCESS){ fprintf(stderr, "compile_topography failed\n"); return 2; }

  cube_t *cube = calloc(1, sizeof(cube_t));
  cube->resolution = res;
  int nprod = 0;
  double t0 = now_ms();
  brick_t **L2 = radiometric_correction(pl2, meta, mission == 0 ? LANDSAT : SENTINEL2, atc, cube, TOA, QAI, AOI, TOP, &nprod);
  double ms = now_ms() - t0;
  if (L2 == NULL){
    wi(o, 1); wd(o, ms); fclose(o);
    return 0;
  }
  wi(o, 0); wd(o, ms);
  dump_atc(o, atc, nb);
  wi(o, nprod);
  for (int k = 0; k < nprod; k++) dump_product(o, L2[k]);
  /* the gaseous transmittance map of every band (Sentinel-2, with the WVP product), from the final QAI */
  if (mission == 1 && fl[2] && fl[11]){
    brick_t *WVP = L2[nprod - 1];
    if (wvp_transmitt_lut(meta, atc) != SUCCESS){ fprintf(stderr, "wvp_transmitt_lut failed\n"); return 2; }
    wi(o, nb);
    for (int b = 0; b < nb; b++){
      short *tg = gas_transmittance(atc, b, WVP, L2[0]);
      fwrite(tg, 2, npix, o);
      free(tg);
    }
  } else wi(o, 0);
  fclose(o);
  return 0;
}

int main(int argc, char **argv){
  if (argc == 4 && strcmp(argv[1], "scene") == 0) return scene(argv[2], argv[3]);
  fprintf(stderr, "usage: l2_harness scene in.bin out.bin\n");
  return 2;
}
