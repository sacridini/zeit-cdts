# RStoolbox's spectral unmixing (mesma, its NNLS solver, without the sum to one it applies afterwards by dividing)
# of an image read from a binary file of float64 (pixels x bands, row-major pixels) with the endmembers of a CSV
# (endmembers x bands): the call timed, the fractions and RMSE written next to the input for the check.
args <- commandArgs(TRUE)
side <- as.integer(args[2]); nb <- as.integer(args[3])
v <- matrix(readBin(args[1], "double", side * side * nb), ncol = nb, byrow = TRUE)
em <- as.matrix(read.csv(args[4], header = FALSE))
suppressMessages({library(terra); library(RStoolbox)})
img <- rast(nrows = side, ncols = side, nlyrs = nb, vals = v)
s <- system.time(out <- mesma(img, em, method = "NNLS", sum_to_one = FALSE, verbose = FALSE))[["elapsed"]]
writeBin(as.vector(t(values(out))), sub(".bin", "_out.bin", args[1], fixed = TRUE))
cat("SECONDS", s, "\n")
