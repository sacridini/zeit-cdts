# remotePARTS (1.0.4) on a map of annual series read from a binary file of float64 (pixels x years, row-major
# pixels, the pixels of a side x side grid row by row): fitAR_map (each pixel's AR(1) trend) and fitGLS_partition
# (the partitioned GLS of the slopes against a mean, exponential correlation of a known range on the grid's
# Euclidean distances, the nugget by maximum likelihood) timed apart. Written next to the input: the slopes, their
# p-values and rhos (_ar.bin), the partitions (_pm.bin, 0-based, one row per partition), the overall estimate, SE,
# t and p (_parts.bin), each partition's estimate, SE and nugget (_part.bin), and the seconds of fitAR_map
# (_ar_seconds.txt); fitGLS_partition's on the SECONDS line.
args <- commandArgs(TRUE)
side <- as.integer(args[2]); nt <- as.integer(args[3]); range <- as.numeric(args[4])
partsize <- as.integer(args[5]); npart <- as.integer(args[6])
suppressMessages(library(remotePARTS))
set.seed(1)
npix <- side * side
Y <- matrix(readBin(args[1], "double", npix * nt), ncol = nt, byrow = TRUE)
k <- seq_len(npix) - 1
coords <- cbind(x = k %% side, y = k %/% side)
out <- function(suffix) sub(".bin", suffix, args[1], fixed = TRUE)

s_ar <- system.time(ar <- fitAR_map(Y, coords))[["elapsed"]]
writeBin(c(ar$coefficients[, 2], ar$pvals[, 2], ar$rhos), out("_ar.bin"))
writeLines(as.character(s_ar), out("_ar_seconds.txt"))

distm_euc <- function(coords, coords2 = NULL) {
  coords <- as.matrix(coords)
  if (is.null(coords2)) return(as.matrix(dist(coords)))
  coords2 <- as.matrix(coords2)
  sqrt(outer(coords[, 1], coords2[, 1], "-")^2 + outer(coords[, 2], coords2[, 2], "-")^2)
}
map <- data.frame(x = coords[, 1], y = coords[, 2], slope = ar$coefficients[, 2])
pm <- sample_partitions(npix = npix, partsize = partsize, npart = npart)
# progressbar = FALSE fails in 1.0.4 ('pb' not found): the bar is drawn and dropped
s <- system.time(invisible(capture.output(f <- suppressWarnings(fitGLS_partition(
  slope ~ 1, partmat = pm, data = map, covar_FUN = "covar_exp", covar.pars = c(range = range),
  distm_FUN = "distm_euc", coord.names = c("x", "y"), nugget = NA, ncross = 6, do.chisqr.test = FALSE)))))[["elapsed"]]
writeBin(as.integer(pm - 1L), out("_pm.bin"))  # column-major partsize x npart: one partition after another
writeBin(as.numeric(f$overall$t.test), out("_parts.bin"))
writeBin(c(f$part$coefficients, f$part$SEs, f$part$nuggets), out("_part.bin"))
cat("SECONDS", s, "\n")
