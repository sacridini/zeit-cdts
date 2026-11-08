/* The harness of the Level 1 comparison (Zeit), compiled against FORCE 3.11.00's own headers and linked with FORCE's own
 * object files (the ones `make` built for `force-l2ps`: meta-ll, read-ll, atc-ll, quality-cl, imagefuns-cl,
 * brick_base-cl, ...), so that `parse_metadata`, `bounds_level1` and `convert_level1` run as the L2PS runs them. Copyright
 * (C) 2013-2022 David Frantz, GPL-3.0-or-later, for what it links; this file is Zeit's, same license.
 *
 * `parse_metadata` is the real one: it reads the product file and the granule file of a (fake) .SAFE directory, finds
 * the 13 band files `*_Bxx.jp2` (GeoTIFFs under that name: GDAL opens them by content; the harness' only use of them
 * is the projection of the first, which FORCE copies to the brick), and fills `meta_t` and the DN brick. `read_level1`
 * is not run (it reads 13 rasters of 10980 x 10980): the DN bands are given and the brick's size is set to theirs
 * (a window of the tile; the geometry of the metadata is compared before that).
 *
 *   l1_harness granule_dir granule_name doreproj dotile bufnodata out.bin [dn.bin]
 *     granule_dir: the directory of the granule (FORCE's d_level1: .../X.SAFE/GRANULE/<name>), granule_name its base name
 *     dn.bin (little endian): int32 nx, ny; uint16 dn[13][nx*ny]
 *     out.bin: int32 status (0 ok, 1 parse_metadata failed, 2 later failure);
 *       int32 nb; per band: int32 rsr_band, float32 radd, float32 scale; char sensor[8]; int32 dtype, sat, tier;
 *       char refsys_id[8]; int32 year, month, day, doy, hh, mm, ss, tz; float64 res, ulx, uly; int32 ncols, nrows;
 *       int32 s2nx, s2ny, left, right, top, bottom; float32 nodata; float32 vzen[s2ny*s2nx], vazi, szen, sazi;
 *       per band: float32 wavelength, char domain[16];
 *       if dn.bin: int32 nx, ny; int16 qai[nx*ny]; int16 toa[13][nx*ny]. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "read-ll.h"
#include "atc-ll.h"
#include "../cross-level/date-cl.h"

/* brdf_factor of brdf-ll.c is not linked (nothing here calls it) */
int brdf_factor(brick_t *sun, brick_t *view, brick_t *cor, int g){ (void)sun; (void)view; (void)cor; (void)g; return SUCCESS; }

static void w32(FILE *f, int v){ fwrite(&v, 4, 1, f); }
static void wf(FILE *f, float v){ fwrite(&v, 4, 1, f); }
static void wd(FILE *f, double v){ fwrite(&v, 8, 1, f); }
static void wstr(FILE *f, const char *s, int n){ char buf[32] = {0}; strncpy(buf, s, n - 1); fwrite(buf, 1, n, f); }

int main(int argc, char **argv){
  if (argc < 7){ fprintf(stderr, "usage: l1_harness granule_dir granule_name doreproj dotile bufnodata out.bin [dn.bin]\n"); return 2; }
  FILE *fo = fopen(argv[6], "wb");
  if (fo == NULL) return 2;

  GDALAllRegister();

  par_ll_t *pl2 = allocate_param_lower();
  pl2->params = allocate_params();
  strcpy(pl2->params->log, "harness"); /* set_brick_par copies it (an empty string is refused) */
  snprintf(pl2->d_level1, NPOW_10, "%s", argv[1]);
  snprintf(pl2->b_level1, NPOW_10, "%s", argv[2]);
  pl2->doreproj = atoi(argv[3]); pl2->dotile = atoi(argv[4]); pl2->bufnodata = atoi(argv[5]);
  pl2->impulse = 0; pl2->nthread = 1; pl2->ithread = 0;
  pl2->res_sentinel2 = 10; pl2->res_landsat = 30;
  default_gdaloptions(_FMT_GTIFF_, &pl2->gdalopt);
  pl2->d_temp = strdup("/tmp"); pl2->d_log = strdup("NULL"); pl2->d_aod = strdup("NULL"); pl2->d_wvp = strdup("NULL");
  pl2->fdem = strdup("NULL");

  meta_t *meta = NULL; brick_t *DN = NULL; int mission = 0;
  if (parse_metadata(pl2, &meta, &DN, &mission) == FAILURE){ w32(fo, 1); fclose(fo); return 0; }
  if (mission != SENTINEL2){ w32(fo, 1); fclose(fo); return 0; }

  int nb = get_brick_nbands(DN);
  w32(fo, 0);
  w32(fo, nb);
  for (int b = 0; b < nb; b++){ w32(fo, meta->cal[b].rsr_band); wf(fo, meta->cal[b].radd); wf(fo, get_brick_scale(DN, b)); }
  char sensor[32]; get_brick_sensor(DN, 0, sensor, 32); wstr(fo, sensor, 8);
  w32(fo, meta->dtype); w32(fo, meta->sat); w32(fo, meta->tier);
  wstr(fo, meta->refsys_id, 8);
  date_t date = get_brick_date(DN, 0);
  w32(fo, date.year); w32(fo, date.month); w32(fo, date.day); w32(fo, date.doy); w32(fo, date.hh); w32(fo, date.mm);
  w32(fo, date.ss); w32(fo, date.tz);
  wd(fo, get_brick_res(DN)); wd(fo, get_brick_ulx(DN)); wd(fo, get_brick_uly(DN));
  w32(fo, get_brick_ncols(DN)); w32(fo, get_brick_nrows(DN));
  int nx_s = meta->s2.nx, ny_s = meta->s2.ny;
  w32(fo, nx_s); w32(fo, ny_s); w32(fo, meta->s2.left); w32(fo, meta->s2.right); w32(fo, meta->s2.top); w32(fo, meta->s2.bottom);
  wf(fo, meta->s2.nodata);
  for (int i = 0; i < ny_s; i++) fwrite(meta->s2.vzen[i], 4, nx_s, fo);
  for (int i = 0; i < ny_s; i++) fwrite(meta->s2.vazi[i], 4, nx_s, fo);
  for (int i = 0; i < ny_s; i++) fwrite(meta->s2.szen[i], 4, nx_s, fo);
  for (int i = 0; i < ny_s; i++) fwrite(meta->s2.sazi[i], 4, nx_s, fo);
  for (int b = 0; b < nb; b++){
    char dom[32]; get_brick_domain(DN, b, dom, 32);
    wf(fo, get_brick_wavelength(DN, b)); wstr(fo, dom, 16);
  }

  if (argc > 7){
    FILE *fi = fopen(argv[7], "rb");
    if (fi == NULL){ fclose(fo); return 2; }
    int nx, ny;
    if (fread(&nx, 4, 1, fi) != 1 || fread(&ny, 4, 1, fi) != 1) return 2;
    int nc = nx * ny;
    /* read_level1 stand-in: the bands, in a brick of the given size */
    set_brick_ncols(DN, nx); set_brick_nrows(DN, ny);
    allocate_brick_bands(DN, nb, nc, _DT_USHORT_);
    ushort **dn = get_bands_ushort(DN);
    for (int b = 0; b < nb; b++) if (fread(dn[b], 2, nc, fi) != (size_t)nc) return 2;
    fclose(fi);

    atc_t *atc = allocate_atc(pl2, meta, DN);
    if (atc == NULL){ fprintf(stderr, "allocate_atc failed\n"); return 2; }
    brick_t *QAI = NULL, *TOA = NULL;
    if (bounds_level1(meta, DN, &QAI, pl2) == FAILURE){ fprintf(stderr, "bounds_level1 failed\n"); return 2; }
    if (convert_level1(meta, mission, atc, DN, &TOA, QAI) != SUCCESS){ fprintf(stderr, "convert_level1 failed\n"); return 2; }
    w32(fo, nx); w32(fo, ny);
    fwrite(get_band_short(QAI, 0), 2, nc, fo);
    short **toa = get_bands_short(TOA);
    for (int b = 0; b < nb; b++) fwrite(toa[b], 2, nc, fo);
  }
  fclose(fo);
  return 0;
}
