/* Stand-ins for FORCE's brick, QAI and allocation helpers, so that FORCE's own functions (copied verbatim into
 * resmerge_harness.c by make_harness.sh) run on plain arrays. One brick_t is both the TOA brick and the QAI brick.
 *
 * FORCE 3.11.00, Copyright (C) 2013-2022 David Frantz, GPL-3.0-or-later; the shims are Zeit's, same license. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include <omp.h>

#define SUCCESS 0
#define FAILURE 1

typedef struct {
  int nx, ny;
  short **toa;                       /* [band][pixel], bands in the order of find_domain() below */
  char *off, *cloud, *shadow;        /* 1 = off / cloud / shadow */
} brick_t;

static int get_brick_ncols(brick_t *b){ return b->nx; }
static int get_brick_nrows(brick_t *b){ return b->ny; }
static int get_brick_ncells(brick_t *b){ return b->nx * b->ny; }
static short **get_bands_short(brick_t *b){ return b->toa; }
static int get_off(brick_t *b, int p){ return b->off[p]; }
static int get_cloud(brick_t *b, int p){ return b->cloud[p]; }
static int get_shadow(brick_t *b, int p){ return b->shadow[p]; }

static int find_domain(brick_t *b, const char *name){
  (void)b;
  if (!strcmp(name, "GREEN"))    return 0;
  if (!strcmp(name, "RED"))      return 1;
  if (!strcmp(name, "BROADNIR")) return 2;
  if (!strcmp(name, "REDEDGE1")) return 3;
  if (!strcmp(name, "REDEDGE2")) return 4;
  if (!strcmp(name, "REDEDGE3")) return 5;
  if (!strcmp(name, "NIR"))      return 6;
  if (!strcmp(name, "SWIR1"))    return 7;
  if (!strcmp(name, "SWIR2"))    return 8;
  return -1;
}

/* alloc-cl.c's helpers, as they behave: zeroed memory; 2D = one allocation per row, 2DC = one contiguous block */
static void alloc(void **ptr, size_t n, size_t size){ *ptr = calloc(n, size); }
static void alloc_2D(void ***ptr, size_t n1, size_t n2, size_t size){
  void **a = calloc(n1, sizeof(void*));
  for (size_t i = 0; i < n1; i++) a[i] = calloc(n2, size);
  *ptr = a;
}
static void alloc_2DC(void ***ptr, size_t n1, size_t n2, size_t size){
  void **a = calloc(n1, sizeof(void*));
  char *block = calloc(n1 * n2, size);
  for (size_t i = 0; i < n1; i++) a[i] = block + i * n2 * size;
  *ptr = a;
}
static void free_2D(void **ptr, size_t n1){ for (size_t i = 0; i < n1; i++) free(ptr[i]); free(ptr); }
static void free_2DC(void **ptr){ free(ptr[0]); free(ptr); }
