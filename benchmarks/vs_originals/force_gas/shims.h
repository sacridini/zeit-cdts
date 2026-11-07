/* What FORCE's gas-ll.c (copied as it is by make_harness.sh, but for its #include "gas-ll.h") needs around it: the
 * minimal brick_t, atc_t, meta_t and par_ll_t with the fields the file reads, and the helpers of the cross-level files
 * it calls that pull in GDAL (the geographic position of a brick, the citation). Functions of FORCE's that matter to the
 * result (get_qai and set_qai, get_off, get_shadow, get_water, set_vaporfill, convert_brick_ji2p, fequal,
 * atc_get_band_reshaped, get_brick_nodata) are copied verbatim by make_harness.sh, as are alloc and alloc_3D (compiled
 * from alloc-cl.c). FORCE 3.11.00, Copyright (C) 2013-2022 David Frantz, GPL-3.0-or-later; the shims are Zeit's,
 * same license. */
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <time.h>
#include <unistd.h>
#include <omp.h>

#include "../cross-level/const-cl.h"
#include "../cross-level/enum-cl.h"
#include "../cross-level/alloc-cl.h"
#include "../cross-level/string-cl.h"
#include "../lower-level/radtran-ll.h"
#include "../lower-level/table-ll.h"

typedef struct {
  int nx, ny, nb;
  double geotran[_GT_LEN_];
  int nodata[8];
  short **vshort;
  small **vsmall;
  float **vfloat;
  double lon, lat;      /* what get_brick_geo would warp to */
  int year, month, day;
} brick_t;

typedef struct { int rsr_band; } cal_t;
typedef struct { cal_t *cal; } meta_t;
typedef struct { char *d_wvp; float wvp; int wvp_strict; } par_ll_t;

typedef struct {
  float cosszen[2], cosvzen[2];
  brick_t *xy_sun, *xy_view, *xy_Tvo, *xy_Tso, *xy_Tg;
  brick_t **xyz_rho_p, **xyz_T, **xyz_s;
} atc_t;

typedef struct {
  float ***val;
  int nb;
  int nw;
  int nm;
} wvp_lut_t;

#define FORCE_NO_DATA_DUMMY 0
static int get_brick_nbands(brick_t *b){ return b->nb; }
static int get_brick_ncols(brick_t *b){ return b->nx; }
static int get_brick_nrows(brick_t *b){ return b->ny; }
static int get_brick_year(brick_t *b, int k){ (void)k; return b->year; }
static int get_brick_month(brick_t *b, int k){ (void)k; return b->month; }
static int get_brick_day(brick_t *b, int k){ (void)k; return b->day; }
static void get_brick_geo(brick_t *b, int j, int i, double *lon, double *lat){ (void)i; (void)j; *lon = b->lon; *lat = b->lat; }
#define cite_me(i) ((void)0)

static short *get_band_short(brick_t *b, int k){ return b->vshort[k]; }
static small *get_band_small(brick_t *b, int k){ return b->vsmall[k]; }
static short **get_bands_short(brick_t *b){ return b->vshort; }
static float *get_band_float(brick_t *b, int k){ return b->vfloat[k]; }

static int G_B_NIR = -1, G_B_VAPOR = -1;
static int find_domain(brick_t *b, const char *name){
  (void)b;
  if (!strcmp(name, "NIR")) return G_B_NIR;
  if (!strcmp(name, "VAPOR")) return G_B_VAPOR;
  return -1;
}

/* copy_brick(QAI, 1, _DT_SHORT_): a new brick of the same grid with one short band of zeros */
static brick_t *copy_brick(brick_t *from, int nb, int datatype){
  (void)datatype;
  brick_t *b = calloc(1, sizeof(brick_t));
  *b = *from;
  b->nb = nb;
  b->vshort = calloc(nb, sizeof(short*));
  for (int k = 0; k < nb; k++) b->vshort[k] = calloc((size_t)from->nx * from->ny, sizeof(short));
  b->vsmall = NULL; b->vfloat = NULL;
  return b;
}
#define set_brick_name(b, s) ((void)0)
#define set_brick_product(b, s) ((void)0)
#define set_brick_filename(b, s) ((void)0)
#define set_brick_bandname(b, k, s) ((void)0)
#define set_brick_nodata(b, k, v) ((void)0)
#define proctime_print(s, t) ((void)0)
