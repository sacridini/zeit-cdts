
/* ---- the harness itself (Zeit) ------------------------------------------------------------------------------
 * radtran_harness <function> <in.bin> <out.bin>
 *   in.bin:  int32 n, then NIN float32 columns of n values each (integers, such as a band, a day or a flag, as whole
 *            floats); out.bin: NOUT float32 columns of n values each. Prints the time of the evaluation (ms). */
static double now_ms(void){
  struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}

static date_t mkdate(float year, float doy){
  date_t d;
  init_date(&d);
  set_date_year(&d, (int)year);
  set_date_doy(&d, (int)doy);
  return d;
}

typedef void (*fn_t)(const float *x, float *y);

static void f_mod(const float *x, float *y){ y[0] = molecular_optical_depth(x[0]); }
static void f_aod_elev_factor(const float *x, float *y){ y[0] = aod_elev_factor(x[0], x[1]); }
static void f_mod_elev_factor(const float *x, float *y){ y[0] = mod_elev_factor(x[0]); }
static void f_mod_elev_scale(const float *x, float *y){ y[0] = mod_elev_scale(x[0], x[1], x[2]); }
static void f_aod_elev_scale(const float *x, float *y){ y[0] = aod_elev_scale(x[0], x[1], x[2]); }
static void f_optical_depth(const float *x, float *y){ y[0] = optical_depth(x[0], x[1]); }
static void f_scatt_transmitt(const float *x, float *y){
  float Ts, Tv, tsd, tss, tvd, tvs;
  y[0] = scatt_transmitt(x[0], x[1], x[2], x[3], x[4], &Ts, &Tv, &tsd, &tss, &tvd, &tvs);
  y[1] = Ts; y[2] = Tv; y[3] = tsd; y[4] = tss; y[5] = tvd; y[6] = tvs;
}
static void f_path_ref(const float *x, float *y){
  y[0] = path_ref(x[0] != 0, x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10]);
}
static void f_sphere_albedo(const float *x, float *y){ y[0] = sphere_albedo(x[0], x[1], x[2]); }
static void f_env_weight_aerosol(const float *x, float *y){ y[0] = env_weight_aerosol(x[0]); }
static void f_env_weight_molecular(const float *x, float *y){ y[0] = env_weight_molecular(x[0]); }
static void f_env_weight(const float *x, float *y){ y[0] = env_weight(x[0], x[1], x[2], x[3]); }
static void f_backscatter(const float *x, float *y){ y[0] = backscatter(x[0], x[1], x[2], x[3]); }
static void f_phase_molecular(const float *x, float *y){ y[0] = phase_molecular(x[0]); }
static void f_phase_aerosol(const float *x, float *y){
  float hg[6] = { x[1], x[2], x[3], x[4], x[5], x[6] };
  y[0] = phase_aerosol(x[0], hg);
}
static void f_wvp_transmitt(const float *x, float *y){ y[0] = wvp_transmitt(x[0], x[1], (int)x[2]); }
static void f_ozone_transmitt(const float *x, float *y){ y[0] = ozone_transmitt(x[0], x[1], (int)x[2]); }
static void f_gas_transmitt(const float *x, float *y){ y[0] = gas_transmitt(x[0], x[1], x[2], x[3]); }
static void f_fresnel(const float *x, float *y){ y[0] = fresnel_reflection(x[0]); }
static void f_illumin(const float *x, float *y){ y[0] = illumin(x[0], x[1], x[2], x[3], x[4], x[5]); }
static void f_c_factor_emp(const float *x, float *y){ y[0] = c_factor_emp((double)x[0], (double)x[1]); }
static void f_c_factor_com(const float *x, float *y){ y[0] = c_factor_com(x[0], x[1], x[2]); }
static void f_f_factor(const float *x, float *y){ y[0] = f_factor(x[0], x[1]); }
static void f_wavelength(const float *x, float *y){ y[0] = wavelength((int)x[0]); }
static void f_e0(const float *x, float *y){ y[0] = E0((int)x[0]); }
static void f_ozone_amount(const float *x, float *y){ y[0] = ozone_amount(x[0], x[1], (int)x[2]); }
static void f_localtime2gmt(const float *x, float *y){   /* year, doy, localtime, lon */
  date_t d = mkdate(x[0], x[1]);
  date_t g = localtime2gmt(&d, (double)x[2], (double)x[3]);
  y[0] = g.year; y[1] = g.month; y[2] = g.day; y[3] = g.doy; y[4] = g.hh; y[5] = g.mm; y[6] = g.ss;
}
static void f_standard_sunzenith(const float *x, float *y){   /* year, doy, tz, lat, lon */
  date_t d = mkdate(x[0], x[1]);
  d.tz = (int)x[2];
  y[0] = (float)standard_sunzenith(&d, (double)x[3], (double)x[4]);
}
static void f_sunpos(const float *x, float *y){   /* lat, lon, year, doy, hh, mm, ss, tz */
  date_t d = mkdate(x[2], x[3]);
  set_time(&d, (int)x[4], (int)x[5], (int)x[6]);
  d.tz = (int)x[7];
  sunpos(x[0], x[1], d, &y[0], &y[1]);
}

static const struct { const char *name; int nin, nout; fn_t f; } TABLE[] = {
  {"molecular_optical_depth", 1, 1, f_mod}, {"aod_elev_factor", 2, 1, f_aod_elev_factor},
  {"mod_elev_factor", 1, 1, f_mod_elev_factor}, {"mod_elev_scale", 3, 1, f_mod_elev_scale},
  {"aod_elev_scale", 3, 1, f_aod_elev_scale}, {"optical_depth", 2, 1, f_optical_depth},
  {"scatt_transmitt", 5, 7, f_scatt_transmitt}, {"path_ref", 11, 1, f_path_ref},
  {"sphere_albedo", 3, 1, f_sphere_albedo}, {"env_weight_aerosol", 1, 1, f_env_weight_aerosol},
  {"env_weight_molecular", 1, 1, f_env_weight_molecular}, {"env_weight", 4, 1, f_env_weight},
  {"backscatter", 4, 1, f_backscatter}, {"phase_molecular", 1, 1, f_phase_molecular},
  {"phase_aerosol", 7, 1, f_phase_aerosol}, {"wvp_transmitt", 3, 1, f_wvp_transmitt},
  {"ozone_transmitt", 3, 1, f_ozone_transmitt}, {"gas_transmitt", 4, 1, f_gas_transmitt},
  {"fresnel_reflection", 1, 1, f_fresnel}, {"illumin", 6, 1, f_illumin},
  {"c_factor_emp", 2, 1, f_c_factor_emp}, {"c_factor_com", 3, 1, f_c_factor_com}, {"f_factor", 2, 1, f_f_factor},
  {"wavelength", 1, 1, f_wavelength}, {"e0", 1, 1, f_e0}, {"ozone_amount", 3, 1, f_ozone_amount},
  {"localtime2gmt", 4, 7, f_localtime2gmt}, {"standard_sunzenith", 5, 1, f_standard_sunzenith},
  {"sunpos", 8, 2, f_sunpos},
};

int main(int argc, char **argv){
  if (argc < 4){ fprintf(stderr, "usage: %s function in.bin out.bin\n", argv[0]); return 2; }
  int k, nt = (int)(sizeof(TABLE) / sizeof(TABLE[0]));
  for (k = 0; k < nt; k++) if (!strcmp(argv[1], TABLE[k].name)) break;
  if (k == nt){ fprintf(stderr, "unknown function %s\n", argv[1]); return 2; }
  int nin = TABLE[k].nin, nout = TABLE[k].nout;
  FILE *f = fopen(argv[2], "rb");
  if (!f){ perror("input"); return 2; }
  int n;
  if (fread(&n, 4, 1, f) != 1) return 2;
  float *in = malloc((size_t)n * nin * sizeof(float)), *out = malloc((size_t)n * nout * sizeof(float));
  if (fread(in, sizeof(float), (size_t)n * nin, f) != (size_t)n * nin) return 2;
  fclose(f);
  float x[16], y[16];
  double t0 = now_ms();
  for (int p = 0; p < n; p++){
    for (int c = 0; c < nin; c++) x[c] = in[(size_t)c * n + p];
    TABLE[k].f(x, y);
    for (int c = 0; c < nout; c++) out[(size_t)c * n + p] = y[c];
  }
  double t1 = now_ms();
  f = fopen(argv[3], "wb");
  fwrite(out, sizeof(float), (size_t)n * nout, f);
  fclose(f);
  printf("%.3f\n", t1 - t0);
  return 0;
}
