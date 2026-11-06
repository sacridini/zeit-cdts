# Brooks et al.'s PolarMetrics 0.1.0 (R package, GPL (>= 3): https://github.com/LandscapeDynamics/PolarMetrics,
# installed from its tarball with R CMD INSTALL) on every pixel of a CSV (pixels x dates), the day of the year of each
# date in another CSV: calc_metrics per pixel, timed, one thread. The metrics, one row per pixel and year (pixel, yr,
# es, ems, ms, lms, ls, s_intv, s_avg, s_sd, s_mag, s_mag_std, ems_mag, lms_mag, a_avg), are written next to the input.
#   Rscript polar.R values.csv doys.csv <samples per cycle> <lcut> <hcut>
suppressPackageStartupMessages(library(PolarMetrics))
args <- commandArgs(TRUE)
v <- as.matrix(read.csv(args[1], header = FALSE)); t <- read.csv(args[2], header = FALSE)[[1]]
spc <- as.integer(args[3]); lcut <- as.numeric(args[4]); hcut <- as.numeric(args[5])
s <- system.time(res <- lapply(seq_len(nrow(v)), function(i) calc_metrics(
  v[i, ], t = t, timing = "from_vecs_integer", yr_type = "cal_yr", spc = spc, lcut = lcut, hcut = hcut,
  return_vecs = FALSE, sin_cos = FALSE)))[["elapsed"]]
out <- do.call(rbind, lapply(seq_along(res), function(i) cbind(pixel = i, res[[i]])))
write.table(out, sub(".csv", "_metrics.csv", args[1], fixed = TRUE), sep = ",", row.names = FALSE, col.names = TRUE)
cat("SECONDS", s, "\n")
