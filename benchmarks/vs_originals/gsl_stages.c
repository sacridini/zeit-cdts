/* reads an n x n double matrix (row-major, n from argv[2]) and writes the stages of gsl_eigen_symmv as raw doubles:
   <out>_tau, _d, _sd (after symmtd), _q (unpacked), _eval, _evec (after gsl_eigen_symmv, unsorted) */
#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>
static void put(const char *out, const char *name, const double *p, size_t n){
  char path[4096]; snprintf(path, sizeof path, "%s_%s", out, name);
  FILE *f = fopen(path, "wb"); fwrite(p, sizeof(double), n, f); fclose(f);
}
int main(int argc, char **argv){
  const char *in = argv[1]; int n = atoi(argv[2]); const char *out = argv[3];
  double *buf = malloc(sizeof(double) * n * n);
  FILE *f = fopen(in, "rb"); if (fread(buf, sizeof(double), (size_t)n * n, f) != (size_t)n * n) return 1; fclose(f);
  gsl_matrix *A = gsl_matrix_alloc(n, n), *Q = gsl_matrix_alloc(n, n), *C = gsl_matrix_alloc(n, n);
  for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) gsl_matrix_set(A, i, j, buf[i * n + j]);
  gsl_matrix_memcpy(C, A);
  gsl_vector *tau = gsl_vector_alloc(n - 1), *d = gsl_vector_alloc(n), *sd = gsl_vector_alloc(n - 1);
  gsl_linalg_symmtd_decomp(A, tau);
  gsl_linalg_symmtd_unpack(A, tau, Q, d, sd);
  put(out, "tau", tau->data, n - 1); put(out, "d", d->data, n); put(out, "sd", sd->data, n - 1); put(out, "q", Q->data, (size_t)n * n);
  gsl_vector *ev = gsl_vector_alloc(n); gsl_matrix *evec = gsl_matrix_alloc(n, n);
  gsl_eigen_symmv_workspace *w = gsl_eigen_symmv_alloc(n);
  gsl_eigen_symmv(C, ev, evec, w);
  put(out, "eval", ev->data, n); put(out, "evec", evec->data, (size_t)n * n);
  return 0;
}
