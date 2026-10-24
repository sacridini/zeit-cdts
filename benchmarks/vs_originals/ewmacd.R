# Brooks' EWMACD v1.3.0 script (Brooks et al. 2014; "EWMACD v1.3.0.r" in "EWMACD v1.3.0.7z" at
# https://vtechworks.lib.vt.edu/items/version/7, no licence stated: run, never copied) on every pixel of a CSV
# (pixels x dates, NA missing), its dates in another CSV (year, doy): the loop over the pixels timed (what its
# rasterEngine runs per pixel), the flags written next to the input for the check. Only the script's function
# definitions are evaluated.
args <- commandArgs(TRUE)
for (e in parse(file = args[1])) {
  if (is.call(e) && (identical(e[[1]], as.name("<-")) || identical(e[[1]], as.name("="))) &&
      is.call(e[[3]]) && identical(e[[3]][[1]], as.name("function"))) {
    eval(e, envir = globalenv())
  }
}
v <- as.matrix(read.csv(args[2], header = FALSE)); d <- read.csv(args[3], header = FALSE)
years <- d[[1]]; doys <- d[[2]]
hb <- max(which(years < 2009))
s <- system.time(flags <- t(apply(v, 1, function(p) EWMACD.pixel.for.rasterEngine(
  myPixel = p, nc = 2, ns = 2, DOYs = doys, Years = years, historybound = hb, xBarLimit1 = 1.5, xBarLimit2 = 20,
  lowthresh = -1e9, lambdasigs = 3, lambda = 0.3, rounding = TRUE, persistence = 1, trainingStart = 2005,
  trainingEnd = 2009, testingEnd = 2013))))[["elapsed"]]
write.table(flags, sub(".csv", "_flags.csv", args[2], fixed = TRUE), sep = ",", row.names = FALSE, col.names = FALSE)
cat("SECONDS", s, "\n")
