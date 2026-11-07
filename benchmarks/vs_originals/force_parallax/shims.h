/* Stand-ins for FORCE's brick, QAI and allocation helpers, so that FORCE's own functions (copied verbatim into
 * parallax_harness.c by make_harness.sh) run on plain arrays. One brick_t is both the TOA brick and the QAI brick.
 *
 * FORCE 3.11.00, Copyright (C) 2013-2022 David Frantz, GPL-3.0-or-later; the shims are Zeit's, same license. */
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include <omp.h>

#define SUCCESS 0
#define FAILURE 1
#define _FORCE_NO_DATA_ -9999

typedef unsigned char small;

typedef struct {
  int nx, ny;
  short *re3, *bnir, *nir;           /* REDEDGE3 (B07), BROADNIR (B08), NIR (B8A) */
  char *off, *sat;                   /* 1 = off / saturated */
} brick_t;

static int get_brick_ncols(brick_t *b){ return b->nx; }
static int get_brick_nrows(brick_t *b){ return b->ny; }
static int get_brick_ncells(brick_t *b){ return b->nx * b->ny; }
static int get_off(brick_t *b, int p){ return b->off[p]; }
static int get_saturation(brick_t *b, int p){ return b->sat[p]; }

static short *get_domain_short(brick_t *b, const char *name){
  if (!strcmp(name, "REDEDGE3")) return b->re3;
  if (!strcmp(name, "BROADNIR")) return b->bnir;
  if (!strcmp(name, "NIR"))      return b->nir;
  return NULL;
}

/* alloc-cl.c's helpers, as they behave: zeroed memory; 2D = one allocation per row */
static void alloc(void **ptr, size_t n, size_t size){ *ptr = calloc(n, size); }
static void alloc_2D(void ***ptr, size_t n1, size_t n2, size_t size){
  void **a = calloc(n1, sizeof(void*));
  for (size_t i = 0; i < n1; i++) a[i] = calloc(n2, size);
  *ptr = a;
}
static void free_2D(void **ptr, size_t n1){ for (size_t i = 0; i < n1; i++) free(ptr[i]); free(ptr); }

/* queue-cl.h */
typedef struct {
  short *buf_x;
  short *buf_y;
  int head;
  int tail;
  int size;
} queue_t;

/* the one line make_harness.sh adds to cloud_parallax: a copy of the CDI (the 20 m grid, float) for the comparison */
static float *dumped_cdi = NULL;
static void harness_dump_cdi(float *cdi, int n){ dumped_cdi = malloc(n * sizeof(float)); memcpy(dumped_cdi, cdi, n * sizeof(float)); }
