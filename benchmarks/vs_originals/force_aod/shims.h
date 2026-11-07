/* What FORCE's aod-ll.c (copied as it is by make_harness.sh, from its start to the end of aod_polynomial_fit, without its
 * first include aod-ll.h) needs around it: the minimal brick_t, atc_t, meta_t, par_ll_t and top_t with the fields the file
 * reads, the types of aod-ll.h (copied verbatim), and the helpers of the cross-level files it calls that pull in GDAL
 * (the geographic position of a brick, the citation). Functions of FORCE's that matter to the result (get_qai, get_off,
 * get_shadow, get_cloud, convert_brick_ji2jip, find_domain, is_brick_nodata, fequal, connectedcomponents_, dist_transform_)
 * are copied verbatim by make_harness.sh, as are alloc and alloc_2D (compiled from alloc-cl.c). FORCE 3.11.00, Copyright
 * (C) 2013-2022 David Frantz, GPL-3.0-or-later; the shims are Zeit's, same license. */
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
#include <gsl/gsl_min.h>

#include "../cross-level/const-cl.h"
#include "../cross-level/enum-cl.h"
#include "../cross-level/alloc-cl.h"
#include "../cross-level/string-cl.h"
#include "../lower-level/radtran-ll.h"
#include "../lower-level/table-ll.h"

typedef struct {
  int nx, ny, nb, nc;
  double geotran[_GT_LEN_];
  int nodata[32];
  short **vshort;
  small **vsmall;
  ushort **vushort;
  float **vfloat;
  char (*domain)[16];
  int doy;
  double lon, lat;      /* what get_brick_geo would warp to */
} brick_t;

typedef struct { int rsr_band; } cal_t;
typedef struct { cal_t *cal; } meta_t;
typedef struct { char *d_aod; bool domulti; bool doaod; char *d_log; } par_ll_t;

typedef struct { brick_t *dem, *exp, *ill, *sky, *c; } top_t;

typedef struct {
  float avg, min, max;
  float step;
  int cnum;
} dem_t;

typedef struct {
  float g1, g2, alpha;
  float hg[6], sob;
} tthg_t;

typedef struct {
  float *wvl, *lwvl, *lwvl2;
  float *E0;
  dem_t dem;
  float Hp, Ha;
  tthg_t tthg;
  bool *aod_bands;
  brick_t *xy_mod, *xy_aod, *xy_Pr, *xy_Pa, *xy_Hr, *xy_Ha, *xy_Tvo, *xy_Tso, *xy_Tg, *xy_brdf, *xy_fresnel, *xy_dem,
          *xy_view, *xy_sun;
} atc_t;

/* the types of aod-ll.h, verbatim */
typedef struct {
  int n;     // number of spectra
  float **s; // spectra
} speclib_t;

typedef struct {
  int k;                      // number of valid pixels
  bool valid;                 // valid object?
  int imin, imax, jmin, jmax; // bounding box
  int i, j;                   // location of centroid
  float ms, mv, razi;         // cos of sun & view, relative azimuth
  int e, f, g;              // coarse grid cell
  float  *ttoa, *etoa;        // target and environment reflectance

  int r;                      // target radius
  float cosi, sky, z;         // target illumination angle, sky view and elevation
  float Ha, Hr;               // elevation correction factors for AOD/MOD
  float  *aod, *est;       // fitted and estimated AOD
  float rsq;                  // R-squared of best fit, and ID of spectrum
  int lib_id;                 // ID of best spectrum
  int ang_fit;                // type of AOD fit; 0: curved, 1: linear
  float coef[3];              // Angstrom coefficients of best fit
} darkobj_t;

typedef struct {
  darkobj_t *wat;       // water targets
  darkobj_t *shd;       // shadow targets
  darkobj_t *veg;       // vegetation targets
  int kwat, kshd, kveg; // number of candidate targets
  int nwat, nshd, nveg; // number of valid targets
} dark_t;

/* the prototypes of aod-ll.h, verbatim (but for the ones of the part of aod-ll.c that is not compiled) */
float *aodfileread(par_ll_t *pl2, atc_t *atc);
int extract_dark_target(atc_t *atc, brick_t *TOA, brick_t *QAI, top_t *TOP, int type, darkobj_t **DOBJ);
int aod_from_target(par_ll_t *pl2, meta_t *meta, atc_t *atc, double res, darkobj_t *dobj, int num, int type);
speclib_t *water_lib(int nb, meta_t *meta);
speclib_t *land_lib(int nb, meta_t *meta);
speclib_t *veg_lib(int nb, int blue, int green, int red);
int aod_lib_to_target(atc_t *atc, double res, bool multi, darkobj_t dobj, speclib_t *lib, int type, float **aodest);
int aod_linear_fit(atc_t *atc, float *logaod, float *angb, float *angn);
int aod_polynomial_fit(atc_t *atc, int naod, float *logaod, float *a0, float *a1, float *a2);
int aod_elev_dependency(atc_t *atc, dark_t *dark, int green);

static int get_brick_nbands(brick_t *b){ return b->nb; }
static int get_brick_ncols(brick_t *b){ return b->nx; }
static int get_brick_nrows(brick_t *b){ return b->ny; }
static int get_brick_ncells(brick_t *b){ return b->nc; }
static double get_brick_res(brick_t *b){ return b->geotran[_GT_RES_]; }
static int get_brick_doy(brick_t *b, int k){ (void)k; return b->doy; }
static void get_brick_geo(brick_t *b, int j, int i, double *lon, double *lat){ (void)i; (void)j; *lon = b->lon; *lat = b->lat; }
static void get_brick_domain(brick_t *b, int k, char *out, int n){ snprintf(out, n, "%s", b->domain[k]); }
#define cite_me(i) ((void)0)
#define proctime_print(s, t) ((void)0)

static short *get_band_short(brick_t *b, int k){ return b->vshort[k]; }
static small *get_band_small(brick_t *b, int k){ return b->vsmall[k]; }
static ushort *get_band_ushort(brick_t *b, int k){ return b->vushort[k]; }
static short **get_bands_short(brick_t *b){ return b->vshort; }
static float get_brick(brick_t *b, int k, int p){ return b->vfloat[k][p]; }
static void set_brick(brick_t *b, int k, int p, float v){ b->vfloat[k][p] = v; }
