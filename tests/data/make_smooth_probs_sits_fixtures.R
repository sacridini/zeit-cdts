# Reference of zeit.smooth_probs: sits' Bayesian smoothing (sits_smooth, sits 1.5.4, GPL-2) of
# seeded probability maps, through sits' own smoothing function of a block (.smooth_fn_bayes:
# the logits, bayes_smoother_fraction, back to probabilities).
#
#   Rscript tests/data/make_smooth_probs_sits_fixtures.R
#
# writes tests/data/smooth_probs_sits.json next to this script.
suppressMessages(library(sits))
library(jsonlite)

smooth_fn_bayes <- get(".smooth_fn_bayes", envir = asNamespace("sits"))

make_probs <- function(nrows, ncols, nclasses, seed) {
  set.seed(seed)
  # smooth class fields (sums of a few bumps) plus noise, softmax per pixel
  rows <- rep(seq_len(nrows), each = ncols)
  cols <- rep(seq_len(ncols), times = nrows)
  scores <- sapply(seq_len(nclasses), function(k) {
    f <- 0
    for (b in 1:3) {
      cy <- runif(1, 1, nrows); cx <- runif(1, 1, ncols); r <- runif(1, 3, 10)
      f <- f + exp(-((rows - cy)^2 + (cols - cx)^2) / (2 * r^2))
    }
    3 * f + rnorm(length(rows), sd = 0.8)
  })
  p <- exp(scores) / rowSums(exp(scores))
  # sits' probabilities come as integers over 10000
  round(p * 10000) / 10000
}

cases <- list(
  list(nrows = 30, ncols = 40, nclasses = 4, seed = 1, window = 5, fraction = 0.5, smoothness = 20),
  list(nrows = 25, ncols = 33, nclasses = 3, seed = 2, window = 9, fraction = 1.0, smoothness = c(5, 20, 80)),
  list(nrows = 12, ncols = 50, nclasses = 5, seed = 3, window = 7, fraction = 0.3, smoothness = 10)
)

out <- lapply(cases, function(cs) {
  p <- make_probs(cs$nrows, cs$ncols, cs$nclasses, cs$seed)
  # some missing pixels (every class), which sits leaves out of their neighbours'
  set.seed(cs$seed + 100)
  miss <- sample(nrow(p), 8)
  p[miss, ] <- NA
  smoothness <- if (length(cs$smoothness) == 1) rep(cs$smoothness, cs$nclasses) else cs$smoothness
  fn <- smooth_fn_bayes(window_size = cs$window, neigh_fraction = cs$fraction, smoothness = smoothness)
  s <- fn(values = p, block = list(nrows = cs$nrows, ncols = cs$ncols))
  list(
    nrows = cs$nrows, ncols = cs$ncols, nclasses = cs$nclasses, window = cs$window,
    neigh_fraction = cs$fraction, smoothness = smoothness,
    # pixels x classes, row-major pixels; NA as null
    probs = unname(p), smoothed = unname(s), missing = sort(miss) - 1
  )
})

path <- file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))),
                  "smooth_probs_sits.json")
writeLines(toJSON(list(sits = as.character(packageVersion("sits")), cases = out),
                  digits = NA, na = "null", matrix = "rowmajor", auto_unbox = TRUE), path)
cat("wrote", path, "\n")
