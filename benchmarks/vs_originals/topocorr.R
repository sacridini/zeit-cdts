# terra's terrain (Horn slope and aspect) and RStoolbox's topCor(method = "C") (RStoolbox 1.0.2.2, GPL (>= 3)), one
# thread, on the files `topocorr.py` writes into <dir>: dem.tif (the DEM), and for "topcor" img.bin (float64, pixels x
# 6 bands, row-major pixels) and sa.bin (float64, pixels x 2: slope and aspect in radians, from zeit, so that the
# correction is compared and not the slopes). The call is timed; the results are written next to the input.
#   Rscript topocorr.R <dir> terrain
#   Rscript topocorr.R <dir> topcor <sun azimuth, radians> <sun zenith, radians> [own]   # "own": R's own slope and aspect
args <- commandArgs(TRUE)
dir <- args[1]; mode <- args[2]
suppressMessages({library(terra); library(RStoolbox)})
try(terraOptions(threads = 1), silent = TRUE)
dem <- rast(file.path(dir, "dem.tif"))
if (mode == "terrain") {
  s <- system.time(out <- terrain(dem, c("slope", "aspect"), unit = "degrees", neighbors = 8))[["elapsed"]]
  writeBin(as.vector(values(out)[, 1]), file.path(dir, "terra_slope.bin"))
  writeBin(as.vector(values(out)[, 2]), file.path(dir, "terra_aspect.bin"))
} else {
  saz <- as.numeric(args[3]); sz <- as.numeric(args[4]); own <- length(args) >= 5 && args[5] == "own"
  npix <- ncell(dem)
  img <- rast(dem, nlyrs = 6)
  values(img) <- matrix(readBin(file.path(dir, "img.bin"), "double", npix * 6), ncol = 6, byrow = TRUE)
  names(img) <- paste0("b", 1:6)
  if (own) {
    topo <- dem
  } else {
    topo <- rast(dem, nlyrs = 2)
    values(topo) <- matrix(readBin(file.path(dir, "sa.bin"), "double", npix * 2), ncol = 2, byrow = TRUE)
    names(topo) <- c("slope", "aspect")
  }
  s <- system.time(out <- topCor(img, dem = topo, solarAngles = c(saz, sz), method = "C"))[["elapsed"]]
  tag <- if (own) "own" else "given"
  writeBin(as.vector(t(values(out))), file.path(dir, paste0("topcor_", tag, ".bin")))
}
cat("SECONDS", s, "\n")
