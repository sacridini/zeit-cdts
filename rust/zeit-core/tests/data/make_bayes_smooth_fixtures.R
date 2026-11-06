# Writes bayes_smooth_ref.json: a synthetic probability cube (3 classes, 60 x 80
# pixels: rectangular regions of a dominant class, logit noise N(0, 1.5), softmax,
# 5% of the pixels with another class dominant) and bayesEO's bayes_smooth()
# (the method of Camara et al. 2024) on it, for three settings.
# Run: Rscript make_bayes_smooth_fixtures.R <output dir>
library(bayesEO)
library(terra)
library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
outdir <- if (length(args) >= 1) args[1] else "."
set.seed(20261106)

nc <- 3L
nr <- 60L
ncol_ <- 80L

# Dominant class per pixel: four rectangles over a background of class 1.
dominant <- matrix(1L, nr, ncol_)
dominant[1:30, 41:80] <- 2L
dominant[31:60, 1:40] <- 3L
dominant[20:45, 25:55] <- 2L
dominant[5:15, 5:30] <- 3L
# 5% of the pixels get another class dominant.
swap <- sample.int(nr * ncol_, round(0.05 * nr * ncol_))
dominant[swap] <- ((dominant[swap] + sample(1:2, length(swap), replace = TRUE) - 1L) %% nc) + 1L

# Logits: +2 for the dominant class, noise N(0, 1.5), softmax over the classes.
cube <- array(0, c(nc, nr, ncol_))
for (k in seq_len(nc)) {
  cube[k, , ] <- ifelse(dominant == k, 2, 0) + rnorm(nr * ncol_, 0, 1.5)
}
for (r in seq_len(nr)) {
  for (cc in seq_len(ncol_)) {
    e <- exp(cube[, r, cc])
    cube[, r, cc] <- e / sum(e)
  }
}

smooth <- function(window, frac, smoothness) {
  layers <- lapply(seq_len(nc), function(k) rast(cube[k, , ], extent = ext(0, ncol_, 0, nr)))
  s <- bayes_smooth(rast(layers), window_size = window, neigh_fraction = frac, smoothness = smoothness)
  out <- array(0, c(nc, nr, ncol_))
  for (k in seq_len(nc)) {
    out[k, , ] <- matrix(values(s[[k]], mat = FALSE), nr, ncol_, byrow = TRUE)
  }
  out
}

settings <- list(
  list(name = "w9_f05_s20", window = 9L, frac = 0.5, smoothness = c(20, 20, 20)),
  list(name = "w9_f05_s10_20_40", window = 9L, frac = 0.5, smoothness = c(10, 20, 40)),
  list(name = "w5_f07_s20", window = 5L, frac = 0.7, smoothness = c(20, 20, 20))
)
cases <- lapply(settings, function(s) {
  s$output <- as.vector(aperm(smooth(s$window, s$frac, s$smoothness), c(3, 2, 1)))
  s
})
# Flat vectors in (class, row, column) order: aperm gives column-major of the
# reversed axes, which is row-major (class, row, column).
result <- list(
  n_classes = nc, nrows = nr, ncols = ncol_,
  input = as.vector(aperm(cube, c(3, 2, 1))),
  cases = cases
)
write(toJSON(result, digits = 10, auto_unbox = TRUE), file.path(outdir, "bayes_smooth_ref.json"))
