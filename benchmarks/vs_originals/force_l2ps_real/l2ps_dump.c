/* force-l2ps up to the cloud detection, with the intermediate results written to a file: what the L2PS has after
 * `bounds_level1` (the initial QAI), `sun_target_view` (the QAI, the sun and view angles), `convert_level1` (the TOA
 * reflectance), `compile_topography` (the QAI, the binned DEM, slope, illumination) and `detect_clouds` (the QAI), to
 * compare with Zeit's (benchmarks/vs_originals/force_l2ps_real.py). The sequence is that of FORCE 3.11.00's
 * `src/main/lower-level/force-l2ps.c` (Copyright (C) 2013-2022 David Frantz, GPL-3.0-or-later), linked with the objects
 * FORCE's `make` built for force-l2ps (the cloud detection is FORCE's own, run, not read); this file is Zeit's, same
 * license.
 *
 *   l2ps_dump image-dir parameter-file out.bin
 *   out.bin (little endian): int32 nx, ny, nb, gnx, gny; float32 dem_avg, dem_min, dem_max, dem_step;
 *     int16 qai_bounds[nc]; int16 qai_sun[nc]; int16 toa[nb][nc]; int16 qai_topo[nc]; uint8 dem[nc]; uint16 slope[nc];
 *     uint16 aspect[nc]; int16 ill[nc]; uint16 sky[nc]; int16 qai_cloud[nc];
 *     float32 sun_zen[g], sun_azi[g], view_zen[g], view_azi[g] (radians, g = gnx * gny coarse cells), float32 cc,
 *     float32 brdf[nb][g]. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "../cross-level/const-cl.h"
#include "../cross-level/utils-cl.h"
#include "../cross-level/string-cl.h"
#include "../cross-level/cite-cl.h"
#include "../cross-level/brick_base-cl.h"
#include "../cross-level/cube-cl.h"
#include "../cross-level/quality-cl.h"
#include "../cross-level/vector-cl.h"
#include "param-ll.h"
#include "meta-ll.h"
#include "cube-ll.h"
#include "atc-ll.h"
#include "read-ll.h"
#include "sunview-ll.h"
#include "topo-ll.h"
#include "cloud-ll.h"
#include "atmo-ll.h"
#include "gdal.h"

static void w32(FILE *f, int v){ fwrite(&v, 4, 1, f); }
static void wf(FILE *f, float v){ fwrite(&v, 4, 1, f); }

int main(int argc, char *argv[]){
  if (argc != 4){ fprintf(stderr, "usage: l2ps_dump image-dir parameter-file out.bin\n"); return 2; }
  int mission;
  par_ll_t *pl2 = allocate_param_lower();
  meta_t *meta = NULL;
  multicube_t *multicube = NULL;
  atc_t *atc = NULL;
  top_t *TOP = NULL;
  brick_t *DN = NULL, *TOA = NULL, *QAI = NULL;

  copy_string(pl2->d_level1, NPOW_10, argv[1]);
  copy_string(pl2->f_par, NPOW_10, argv[2]);
  CPLPushErrorHandler(CPLQuietErrorHandler);
  GDALAllRegister();
  GDALDriverH driver;
  if ((driver = GDALGetDriverByName("JP2ECW")) != NULL) GDALDeregisterDriver(driver);
  if (parse_param_lower(pl2) != SUCCESS){ printf("Parsing parameter file failed.\n"); return FAILURE; }
  if (parse_metadata(pl2, &meta, &DN, &mission) == FAILURE){ printf("Parsing metadata failed.\n"); return FAILURE; }
  if ((multicube = start_multicube(pl2, DN)) == NULL){ printf("Starting datacube(s) failed.\n"); return FAILURE; }
  omp_set_num_threads(pl2->nthread);
  omp_set_nested(true);
  omp_set_max_active_levels(2);

  FILE *o = fopen(argv[3], "wb");
  if (o == NULL) return 2;

  if (read_level1(meta, mission, DN, pl2) != SUCCESS){ printf("reading DNs failed.\n"); return FAILURE; }
  if ((atc = allocate_atc(pl2, meta, DN)) == NULL){ printf("Allocating atc failed.\n"); return FAILURE; }
  if (bounds_level1(meta, DN, &QAI, pl2) == FAILURE){ printf("Compiling nodata / saturation masks failed.\n"); return FAILURE; }
  int nx = get_brick_ncols(DN), ny = get_brick_nrows(DN), nb = get_brick_nbands(DN), nc = nx * ny;
  int gnx = get_brick_ncols(atc->xy_sun), gny = get_brick_nrows(atc->xy_sun), gn = gnx * gny;
  w32(o, nx); w32(o, ny); w32(o, nb); w32(o, gnx); w32(o, gny);
  long hdr = ftell(o);
  for (int k = 0; k < 4; k++) wf(o, 0.0f);  /* the DEM statistics, written at the end */
  fwrite(get_band_short(QAI, 0), 2, nc, o);

  if (sun_target_view(pl2, meta, mission, atc, QAI) == FAILURE){ printf("computing sun/view geometry failed.\n"); return FAILURE; }
  fwrite(get_band_short(QAI, 0), 2, nc, o);

  if (convert_level1(meta, mission, atc, DN, &TOA, QAI) != SUCCESS){ printf("DN to TOA conversion failed.\n"); return FAILURE; }
  for (int b = 0; b < nb; b++) fwrite(get_band_short(TOA, b), 2, nc, o);

  if (compile_topography(pl2, atc, &TOP, QAI) != SUCCESS){ printf("unable to compile topography.\n"); return FAILURE; }
  fwrite(get_band_short(QAI, 0), 2, nc, o);
  fwrite(get_band_small(TOP->dem, 0), 1, nc, o);
  fwrite(get_band_ushort(TOP->exp, 0), 2, nc, o);
  fwrite(get_band_ushort(TOP->exp, 1), 2, nc, o);
  fwrite(get_band_short(TOP->ill, 0), 2, nc, o);
  fwrite(get_band_ushort(TOP->sky, 0), 2, nc, o);

  /* the TOA before the cloud detection, to see whether the detection changes it */
  short **toa_before = malloc(nb * sizeof(short *));
  for (int b = 0; b < nb; b++){ toa_before[b] = malloc((size_t)nc * 2); memcpy(toa_before[b], get_band_short(TOA, b), (size_t)nc * 2); }
  int err = detect_clouds(pl2, mission, atc, TOA, TOP->dem, TOP->exp, QAI);
  if (err == FAILURE){ printf("error in cloud module.\n"); return FAILURE; }
  for (int b = 0; b < nb; b++){
    short *t = get_band_short(TOA, b);
    long nd = 0; int mx = 0;
    for (int p = 0; p < nc; p++) if (t[p] != toa_before[b][p]){ nd++; int d = abs(t[p] - toa_before[b][p]); if (d > mx) mx = d; }
    printf("TOA band %d changed by the cloud detection: %ld values, max %d\n", b, nd, mx);
  }
  fwrite(get_band_short(QAI, 0), 2, nc, o);
  for (int g = 0; g < gn; g++) wf(o, get_brick(atc->xy_sun, ZEN, g));
  for (int g = 0; g < gn; g++) wf(o, get_brick(atc->xy_sun, AZI, g));
  for (int g = 0; g < gn; g++) wf(o, get_brick(atc->xy_view, ZEN, g));
  for (int g = 0; g < gn; g++) wf(o, get_brick(atc->xy_view, AZI, g));
  wf(o, atc->cc);
  /* FORCE's BRDF factor of every band and coarse cell, as recorded above */
  /* FORCE's BRDF factor of every band and coarse cell: `brdf_factor` (brdf-ll.o, run, not read) called as `atmo_angledep` calls
   * it, once for every cell that has a view angle, on the geometry the correction used (it is the same cells, the same sun and
   * view bricks). It is called on a copy of the brick, so that the correction below is not touched. */
  brick_t *bf = copy_brick(atc->xy_brdf, nb, _DT_FLOAT_);
  for (int g = 0; g < gn; g++){
    if (is_brick_nodata(atc->xy_view, 0, g)) continue;
    brdf_factor(atc->xy_sun, atc->xy_view, bf, g);
  }
  for (int b = 0; b < nb; b++) for (int g = 0; g < gn; g++) wf(o, get_brick(bf, b, g));
  /* the rest of the L2PS: what radiometric_correction prints (the AOD and the number of targets) */
  int nprod = 0;
  brick_t **LEVEL2 = radiometric_correction(pl2, meta, mission, atc, multicube->cube[0], TOA, QAI, NULL, TOP, &nprod);
  printf("\nradiometric_correction %s\n", LEVEL2 == NULL ? "FAILED" : "ok");
  fseek(o, hdr, SEEK_SET);
  wf(o, atc->dem.avg); wf(o, atc->dem.min); wf(o, atc->dem.max); wf(o, atc->dem.step);
  fclose(o);
  printf("done\n");
  return SUCCESS;
}
