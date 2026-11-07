/* What FORCE's topo-ll.c (copied as it is by make_harness.sh, whole, without its first include topo-ll.h) needs around
 * it: the minimal brick_t, atc_t and par_ll_t with the fields the file reads, top_t and the prototypes of topo-ll.h
 * (verbatim), and the brick accessors that pull in GDAL or the whole brick machinery (shims). The functions of the
 * cross-level files that matter to the result (get_qai, get_off, set_qai, set_off, set_slope, set_illumination,
 * convert_brick_p2p, convert_brick_ji2p, convert_brick_p2ji, find_domain, fequal, set_brick, covar_recurrence, variance,
 * covariance, linreg_slope, linreg_intercept, linreg_coefs, atc_get_band_reshaped) are copied verbatim by
 * make_harness.sh, as are alloc (compiled from alloc-cl.c) and the radiative transfer functions (radtran-ll.c). FORCE
 * 3.11.00, Copyright (C) 2013-2022 David Frantz, GPL-3.0-or-later; the shims are Zeit's, same license. */
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
#include "../lower-level/radtran-ll.h"

typedef struct {
  int nx, ny, nb, nc;
  int datatype;
  double geotran[_GT_LEN_];
  int nodata[32];
  short **vshort;
  small **vsmall;
  ushort **vushort;
  float **vfloat;
  int **vint;
  char (*domain)[16];
} brick_t;

typedef struct {
  char *fdem;
  int dem_nodata;
  int dem_resample;
  int nthread;
} par_ll_t;

typedef struct {
  float avg, min, max;
  float step;
  int cnum;
} dem_t;

typedef struct {
  dem_t dem;
  brick_t *xy_sun;
  brick_t **xyz_rho_p, **xyz_tss, **xyz_tsd;
  brick_t *xy_dem;
} atc_t;

/* topo-ll.h, verbatim */
typedef struct {
  brick_t *dem; // binned digital elevation model
  brick_t *exp; // exposition (slope / aspect)
  brick_t *ill; // illumination angle
  brick_t *sky; // sky view factor
  brick_t *c;   // C-factor SWIR2
} top_t;

void free_topography(top_t *top);
int compile_topography(par_ll_t *pl2, atc_t *atc, top_t **topography, brick_t *QAI);
brick_t *cfactor_topography(atc_t *atc, brick_t *TOA, brick_t *QAI, brick_t *DEM, brick_t *EXP, brick_t *ILL);
int average_elevation_cell(int g, brick_t *CDEM, brick_t *FDEM, brick_t *QAI);

/* the cross-level functions copied verbatim (prototypes) */
bool fequal(float a, float b);
short get_qai(brick_t *qai, int index, int p, int bitfields);
bool get_off(brick_t *qai, int p);
void set_qai(brick_t *qai, int index, int p, short val, int bitfields);
void set_off(brick_t *qai, int p, short val);
void set_slope(brick_t *qai, int p, short val);
void set_illumination(brick_t *qai, int p, short val);
int convert_brick_p2p(brick_t *from, brick_t *to, int p_from);
int convert_brick_ji2p(brick_t *from, brick_t *to, int i_from, int j_from);
void convert_brick_p2ji(brick_t *from, brick_t *to, int p_from, int *i_to, int *j_to);
int find_domain(brick_t *brick, const char *domain);
void set_brick(brick_t *brick, int b, int p, float val);
void covar_recurrence(double x, double y, double *mx, double *my, double *vx, double *vy, double *cv, double n);
double variance(double var, double n);
double covariance(double cov, double n);
void linreg_slope(double cov, double varx, double *slope);
void linreg_intercept(double slope, double mx, double my, double *intercept);
void linreg_coefs(double mx, double my, double cov, double varx, double *slope, double *intercept);
float **atc_get_band_reshaped(brick_t **xyz, int b);

static int get_brick_nbands(brick_t *b){ return b->nb; }
static int get_brick_ncols(brick_t *b){ return b->nx; }
static int get_brick_nrows(brick_t *b){ return b->ny; }
static int get_brick_ncells(brick_t *b){ return b->nc; }
static double get_brick_res(brick_t *b){ return b->geotran[_GT_RES_]; }
static int get_brick_nodata(brick_t *b, int k){ return b->nodata[k]; }
static void set_brick_nodata(brick_t *b, int k, int v){ b->nodata[k] = v; }
static void get_brick_domain(brick_t *b, int k, char *out, int n){ snprintf(out, n, "%s", b->domain[k]); }
static void set_brick_name(brick_t *b, const char *s){ (void)b; (void)s; }
static void set_brick_product(brick_t *b, const char *s){ (void)b; (void)s; }
static void set_brick_filename(brick_t *b, const char *s){ (void)b; (void)s; }
static void set_brick_bandname(brick_t *b, int k, const char *s){ (void)b; (void)k; (void)s; }
#define cite_me(i) ((void)0)
#define proctime_print(s, t) ((void)0)

static short *get_band_short(brick_t *b, int k){ return b->vshort[k]; }
static small *get_band_small(brick_t *b, int k){ return b->vsmall[k]; }
static ushort *get_band_ushort(brick_t *b, int k){ return b->vushort[k]; }
static float *get_band_float(brick_t *b, int k){ return b->vfloat[k]; }
static float **get_bands_float(brick_t *b){ return b->vfloat; }
static short *get_domain_short(brick_t *b, const char *domain){
  int k = find_domain(b, domain);
  return (k < 0) ? NULL : b->vshort[k];
}

/* copy_brick: the geometry of `from`, `nb` bands of datatype `dt` (zero-initialized as FORCE's) */
static brick_t *copy_brick(brick_t *from, int nb, int dt){
  brick_t *b = calloc(1, sizeof(brick_t));
  *b = *from;
  b->nb = nb; b->datatype = dt;
  b->vshort = NULL; b->vsmall = NULL; b->vushort = NULL; b->vfloat = NULL; b->vint = NULL;
  b->domain = calloc(nb > 0 ? nb : 1, 16);
  size_t n = (size_t)b->nc;
  if (dt == _DT_SHORT_){ b->vshort = calloc(nb, sizeof(short*)); for (int k = 0; k < nb; k++) b->vshort[k] = calloc(n, sizeof(short)); }
  if (dt == _DT_SMALL_){ b->vsmall = calloc(nb, sizeof(small*)); for (int k = 0; k < nb; k++) b->vsmall[k] = calloc(n, 1); }
  if (dt == _DT_FLOAT_){ b->vfloat = calloc(nb, sizeof(float*)); for (int k = 0; k < nb; k++) b->vfloat[k] = calloc(n, sizeof(float)); }
  if (dt == _DT_INT_){ b->vint = calloc(nb, sizeof(int*)); for (int k = 0; k < nb; k++) b->vint[k] = calloc(n, sizeof(int)); }
  if (dt == _DT_USHORT_){ b->vushort = calloc(nb, sizeof(ushort*)); for (int k = 0; k < nb; k++) b->vushort[k] = calloc(n, sizeof(ushort)); }
  return b;
}
static void free_brick(brick_t *b){ (void)b; /* the harness keeps them: it dumps them after the call */ }

/* the warp of the DEM file (GDAL) is replaced by the DEM the harness holds: the same on both sides */
static float *g_dem_in = NULL;
static int warp_from_disc_to_known_brick(int rsm, int threads, const char *fname, brick_t *dst, int src_b, int dst_b, int src_nodata){
  (void)rsm; (void)threads; (void)fname; (void)src_b; (void)src_nodata;
  memcpy(dst->vfloat[dst_b], g_dem_in, (size_t)dst->nc * sizeof(float));
  return SUCCESS;
}
