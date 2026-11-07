/* What FORCE's atc-ll.c, sunview-ll.c (both copied as they are by make_harness.sh, whole, without their first
 * includes) and the functions interpolation_weights, interpolate_coarse, atmo_angledep, atmo_elevdep and apply_aoi of
 * atmo-ll.c (copied verbatim) need around them: a brick_t with the fields they read and the brick functions that
 * allocate and copy (shims that do what brick_base-cl.c does with them: copy_brick copies the band metadata of band b to
 * band b if both bricks have as many bands, of band 0 to every band otherwise), the structures meta_t, par_ll_t, top_t
 * and the shim of the sun position (the table the harness holds, see main.inc) and of the warp of the map coordinates
 * to longitude and latitude (an affine map, the same on both sides). The types of atc-ll.h, the s2_meta and cal_t of
 * meta-ll.h, top_t of topo-ll.h and iweights_t of atmo-ll.c are copied by make_harness.sh.
 * FORCE 3.11.00, Copyright (C) 2013-2022 David Frantz, GPL-3.0-or-later; the shims are Zeit's, same license. */
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
#include "../cross-level/date-cl.h"
#include "../lower-level/radtran-ll.h"

typedef struct { char *string; } hstr_t;

typedef struct {
  int nx, ny, nb, nc;
  int datatype;
  double width, height;
  double geotran[_GT_LEN_];
  int *nodata;
  float *wavelength;
  date_t *date;
  char (*domain)[16];
  hstr_t proj;
  short **vshort;
  small **vsmall;
  ushort **vushort;
  float **vfloat;
  int **vint;
} brick_t;

/* meta-ll.h, verbatim (but for meta_t, which keeps only what the files read) */
typedef struct {
  int   rsr_band;       // ID in RSR table
} cal_t;

typedef struct {
  float nodata;
  float **szen, **sazi; // sun  zenith / azimuth from metadata
  float **vzen, **vazi; // view zenith / azimuth from metadata
  int nx, ny;           // number of cells in angle grid
  int left, right;
  int top, bottom;
} s2_meta;

typedef struct {
  cal_t *cal;
  s2_meta s2;
} meta_t;

typedef struct {
  bool domulti;
  bool doenv;
} par_ll_t;

/* topo-ll.h, verbatim */
typedef struct {
  brick_t *dem; // binned digital elevation model
  brick_t *exp; // exposition (slope / aspect)
  brick_t *ill; // illumination angle
  brick_t *sky; // sky view factor
  brick_t *c;   // C-factor SWIR2
} top_t;

/*@ATC_H@*/

/* atmo-ll.c, verbatim */
/*@IWEIGHTS@*/

/* prototypes */
atc_t *allocate_atc(par_ll_t *pl2, meta_t *meta, brick_t *DN);
float **atc_get_band_reshaped(brick_t **xyz, int b);
double standard_sunzenith(date_t *dmeta, double lat, double lon);
int sun_target_view(par_ll_t *pl2, meta_t *meta, int mission, atc_t *atc, brick_t *QAI);
int viewgeo(par_ll_t *pl2, brick_t *QAI, atc_t *atc);
int view_angle(meta_t *meta, int mission, atc_t *atc, brick_t *QAI, int f, int e, int g);
iweights_t interpolation_weights(int j, int i, int nf, int ne, double res, double full_res, float *COARSE, float nodata);
float interpolate_coarse(iweights_t weight, float *COARSE);
int atmo_angledep(par_ll_t *pl2, meta_t *meta, atc_t *atc, top_t *TOP, brick_t *QAI);
int atmo_elevdep(par_ll_t *pl2, atc_t *atc, brick_t *QAI, top_t *TOP);
int apply_aoi(brick_t *QAI, brick_t *AOI);
int average_elevation_cell(int g, brick_t *CDEM, brick_t *FDEM, brick_t *QAI);
float ozone_amount(float lon, float lat, int doy);
bool fequal(float a, float b);
short get_qai(brick_t *qai, int index, int p, int bitfields);
void set_qai(brick_t *qai, int index, int p, short val, int bitfields);
bool get_off(brick_t *qai, int p);
void set_off(brick_t *qai, int p, short val);
void set_lowsun(brick_t *qai, int p, short val);
void set_aerosol(brick_t *qai, int p, short val);
int convert_brick_p2p(brick_t *from, brick_t *to, int p_from);
void convert_brick_p2ji(brick_t *from, brick_t *to, int p_from, int *i_to, int *j_to);
int find_domain(brick_t *brick, const char *domain);
bool is_brick_nodata(brick_t *brick, int b, int p);
void set_brick(brick_t *brick, int b, int p, float val);
float get_brick(brick_t *brick, int b, int p);
void get_brick_range(brick_t *brick, int b, float *min, float *max);
double get_brick_x(brick_t *brick, int j);
double get_brick_y(brick_t *brick, int i);
void get_brick_geo(brick_t *brick, int j, int i, double *lon, double *lat);
date_t get_brick_date(brick_t *brick, int b);
int get_brick_doy(brick_t *brick, int b);

/* the brick accessors, as brick_base-cl.c has them */
static int get_brick_nbands(brick_t *b){ return b->nb; }
static int get_brick_ncols(brick_t *b){ return b->nx; }
static int get_brick_nrows(brick_t *b){ return b->ny; }
static int get_brick_ncells(brick_t *b){ return b->nc; }
static double get_brick_width(brick_t *b){ return b->width; }
static double get_brick_height(brick_t *b){ return b->height; }
static double get_brick_res(brick_t *b){ return b->geotran[_GT_RES_]; }
static float get_brick_wavelength(brick_t *b, int k){ return b->wavelength[k]; }
static void get_brick_domain(brick_t *b, int k, char *out, int n){ snprintf(out, n, "%s", b->domain[k]); }
static small *get_band_small(brick_t *b, int k){ return b->vsmall[k]; }
static float *get_band_float(brick_t *b, int k){ return b->vfloat[k]; }
static void set_brick_name(brick_t *b, const char *s){ (void)b; (void)s; }
static void set_brick_product(brick_t *b, const char *s){ (void)b; (void)s; }
static void set_brick_filename(brick_t *b, const char *s){ (void)b; (void)s; }
static void set_brick_nodata(brick_t *b, int k, int v){ b->nodata[k] = v; }
static void set_brick_ncols(brick_t *b, int nx){ b->nx = nx; b->nc = b->nx * b->ny; b->width = b->nx * b->geotran[_GT_RES_]; }
static void set_brick_nrows(brick_t *b, int ny){ b->ny = ny; b->nc = b->nx * b->ny; b->height = b->ny * b->geotran[_GT_RES_]; }
static void set_brick_res(brick_t *b, double res){
  b->geotran[_GT_XRES_] = res; b->geotran[_GT_YRES_] = res * -1;
  b->width = b->nx * res; b->height = b->ny * res;
}
#define cite_me(i) ((void)0)
#define proctime_print(s, t) ((void)0)
static void free_brick(brick_t *b){ (void)b; /* the harness keeps them: it dumps them after the call */ }

static void alloc_brick_meta(brick_t *b, int nb){
  b->nb = nb;
  b->nodata = calloc(nb, sizeof(int));
  b->wavelength = calloc(nb, sizeof(float));
  b->date = calloc(nb, sizeof(date_t));
  b->domain = calloc(nb, 16);
}

/* allocate_brick_bands (brick_base-cl.c): the bands of nc cells, zero-initialized */
static int allocate_brick_bands(brick_t *b, int nb, int nc, int dt){
  b->datatype = dt;
  if (dt == _DT_SHORT_){ b->vshort = calloc(nb, sizeof(short*)); for (int k = 0; k < nb; k++) b->vshort[k] = calloc(nc, sizeof(short)); }
  if (dt == _DT_SMALL_){ b->vsmall = calloc(nb, sizeof(small*)); for (int k = 0; k < nb; k++) b->vsmall[k] = calloc(nc, 1); }
  if (dt == _DT_FLOAT_){ b->vfloat = calloc(nb, sizeof(float*)); for (int k = 0; k < nb; k++) b->vfloat[k] = calloc(nc, sizeof(float)); }
  if (dt == _DT_INT_){ b->vint = calloc(nb, sizeof(int*)); for (int k = 0; k < nb; k++) b->vint[k] = calloc(nc, sizeof(int)); }
  if (dt == _DT_USHORT_){ b->vushort = calloc(nb, sizeof(ushort*)); for (int k = 0; k < nb; k++) b->vushort[k] = calloc(nc, sizeof(ushort)); }
  return SUCCESS;
}

/* copy_brick (brick_base-cl.c): the bricks `nb` bands of `dt` with the geometry of `from`, and its band metadata: of
 * band b to band b if from has nb bands, of its band 0 to every band otherwise */
static brick_t *copy_brick(brick_t *from, int nb, int dt){
  brick_t *b = calloc(1, sizeof(brick_t));
  alloc_brick_meta(b, nb);
  memcpy(b->geotran, from->geotran, sizeof(b->geotran));
  b->nx = from->nx; b->ny = from->ny; b->nc = b->nx * b->ny;
  b->width = b->nx * b->geotran[_GT_RES_]; b->height = b->ny * b->geotran[_GT_RES_];
  for (int k = 0; k < nb; k++){
    int kf = (nb == from->nb) ? k : 0;
    b->nodata[k] = from->nodata[kf];
    b->wavelength[k] = from->wavelength[kf];
    b->date[k] = from->date[kf];
    memcpy(b->domain[k], from->domain[kf], 16);
  }
  allocate_brick_bands(b, nb, from->nc, dt);
  return b;
}

/* the map to longitude and latitude (warp_any_to_geo of warp-cl.cpp, GDAL): an affine map, set by the harness */
static double g_lon0 = 0, g_x0 = 0, g_kgeo = 1;
static int warp_any_to_geo(double srs_x, double srs_y, double *dst_x, double *dst_y, char *src_wkt){
  (void)src_wkt;
  *dst_x = g_lon0 + (srs_x - g_x0) * g_kgeo;
  *dst_y = srs_y * g_kgeo;
  return SUCCESS;
}

/* the sun position (sun-cl.c is not opened, see Zeit's solpos.rs): the harness holds a table of the values Zeit's
 * gives for the places of the coarse grid, looked up by the exact latitude and longitude */
typedef struct { double lat, lon; float zen, azi; } sun_entry_t;
static sun_entry_t *g_sun = NULL;
static int g_nsun = 0;
static void sunpos(double lat, double lon, date_t date, float *zen, float *azi){
  (void)date;
  for (int k = 0; k < g_nsun; k++){
    if (g_sun[k].lat == lat && g_sun[k].lon == lon){ *zen = g_sun[k].zen; *azi = g_sun[k].azi; return; }
  }
  fprintf(stderr, "sunpos: no entry for lat %.17g lon %.17g\n", lat, lon);
  exit(3);
}
#define print_date(d) ((void)0)

/* brdf_factor (brdf-ll.c) is not ported (nor opened): the factor FORCE starts from, 1 for every band */
static int brdf_factor(brick_t *sun, brick_t *view, brick_t *cor, int g){
  (void)sun; (void)view;
  for (int b = 0; b < get_brick_nbands(cor); b++) set_brick(cor, b, g, 1.0);
  return SUCCESS;
}
