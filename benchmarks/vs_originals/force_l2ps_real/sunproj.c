/* What force-l2ps uses for the geography and the sun of the coarse grid, as FORCE computes them: the longitude and latitude
 * of the corners of the cells (`warp_any_to_geo` of warp-cl.c with the projection of the first band file, which is what
 * `get_brick_geo` calls) and the sun zenith and azimuth of each (`sunpos` of sun-cl.c, called as `sun_target_view` calls it:
 * with the latitude and longitude as floats and the date of the first band). The comparison with FORCE gives these to Zeit
 * (Zeit's sun is a clean-room implementation and its projection is rasterio's, not the C's), to separate the effect of the
 * sun and of the geography from the others. Compiled against FORCE 3.11.00's headers (GPL-3.0-or-later, Copyright (C)
 * 2013-2022 David Frantz) and linked with the objects FORCE's `make` built; this file is Zeit's, same license.
 *
 *   sunproj band_file year month day hh mm ss n ulx uly res out.bin
 *   out.bin: float64 lon[n*n], float64 lat[n*n], float32 zen[n*n], float32 azi[n*n] (row by row; the corner of column j, row i
 *   is (ulx + j * res, uly + i * -res)) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gdal.h"
#include "../cross-level/date-cl.h"
#include "../cross-level/sun-cl.h"
#include "../cross-level/warp-cl.h"

/* brdf-ll.o is not linked */
struct brick_s;
int brdf_factor(void *sun, void *view, void *cor, int g){ (void)sun; (void)view; (void)cor; (void)g; return 0; }

int main(int argc, char **argv){
  if (argc < 13){ fprintf(stderr, "usage: sunproj band_file year month day hh mm ss n ulx uly res out.bin\n"); return 2; }
  GDALAllRegister();
  GDALDatasetH ds = GDALOpen(argv[1], GA_ReadOnly);
  if (ds == NULL){ fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }
  char wkt[10000];
  snprintf(wkt, sizeof(wkt), "%s", GDALGetProjectionRef(ds));
  GDALClose(ds);

  date_t date;
  init_date(&date);
  set_date(&date, atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
  date.hh = atoi(argv[5]); date.mm = atoi(argv[6]); date.ss = atoi(argv[7]); date.tz = 0;
  int n = atoi(argv[8]);
  double ulx = atof(argv[9]), uly = atof(argv[10]), res = atof(argv[11]);

  size_t nc = (size_t)n * n;
  double *lon = malloc(nc * 8), *lat = malloc(nc * 8);
  float *zen = malloc(nc * 4), *azi = malloc(nc * 4);
  for (int i = 0; i < n; i++){
    for (int j = 0; j < n; j++){
      size_t k = (size_t)i * n + j;
      double x = ulx + j * res, y = uly + i * -res;
      if (warp_any_to_geo(x, y, &lon[k], &lat[k], wkt) != 0){ fprintf(stderr, "warp failed\n"); return 3; }
      sunpos((float)lat[k], (float)lon[k], date, &zen[k], &azi[k]);
    }
  }
  FILE *f = fopen(argv[12], "wb");
  fwrite(lon, 8, nc, f); fwrite(lat, 8, nc, f); fwrite(zen, 4, nc, f); fwrite(azi, 4, nc, f);
  fclose(f);
  return 0;
}
