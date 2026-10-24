# RecoverMetricsR's recovery metrics (compute_recovery_metrics: RRI, R80P, YrYr; compute_recovery_time: the years to
# 80 % of the pre-disturbance mean) of an annual series read from a binary file of float64 (pixels x years, row-major
# pixels), every pixel disturbed at the same year: the two calls timed, their layers written next to the input.
args <- commandArgs(TRUE)
side <- as.integer(args[2]); first <- as.integer(args[3]); nt <- as.integer(args[4]); event <- as.integer(args[5])
v <- matrix(readBin(args[1], "double", side * side * nt), ncol = nt, byrow = TRUE)
suppressMessages({library(terra); library(RecoverMetricsR)})
img <- rast(nrows = side, ncols = side, nlyrs = nt, vals = v)
years <- first:(first + nt - 1)
s <- system.time({
  m <- compute_recovery_metrics(img, years, event, pre_window = 2, post_window = 5, metrics = c("RRI", "R80P", "YrYr"),
                                clamp = FALSE)
  ttr <- compute_recovery_time(img, years, event, proportion = 0.8, pre_window = 2)
})[["elapsed"]]
writeBin(as.vector(t(values(c(m, ttr)))), sub(".bin", "_out.bin", args[1], fixed = TRUE))
cat("SECONDS", s, "\n")
