# RStoolbox's tasseledCap (RStoolbox 1.0.2.2, GPL (>= 3)) on an image read from a binary file of float64 (pixels x
# bands, row-major pixels), for the sensor of the 4th argument (landsat5tm, landsat7etm, landsat8oli): the call timed, the
# three components written next to the input (pixels x 3, float64) for the check. One thread (terra's app).
#   Rscript tasseled.R img.bin <side> <sat> <suffix>
args <- commandArgs(TRUE)
side <- as.integer(args[2]); sat <- args[3]; suffix <- args[4]
v <- matrix(readBin(args[1], "double", side * side * 6), ncol = 6, byrow = TRUE)
suppressMessages({library(terra); library(RStoolbox)})
img <- rast(nrows = side, ncols = side, nlyrs = 6, vals = v)
s <- system.time(out <- tasseledCap(img, sat))[["elapsed"]]
writeBin(as.vector(t(values(out))), sub(".bin", paste0("_", suffix, ".bin"), args[1], fixed = TRUE))
cat("SECONDS", s, "\n")
