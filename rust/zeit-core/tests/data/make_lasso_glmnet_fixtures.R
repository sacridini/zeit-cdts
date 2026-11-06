# Writes lasso_glmnet.json: synthetic CCDC-like cases and glmnet's coefficients
# (alpha = 1, standardize = TRUE, intercept = TRUE, lambda = 20) at thresh 1e-4
# and 1e-12. Run: Rscript make_lasso_glmnet_fixtures.R <output dir>
library(glmnet)
library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
outdir <- if (length(args) >= 1) args[1] else "."
set.seed(20261105)

make_case <- function(id, n, df, const_col) {
  harmonics <- (df - 2) %/% 2
  days <- 730000 + sort(round(runif(n, 0, 16 * n * 1.2)))
  w <- 2 * pi / 365.25
  cols <- list(days)
  for (h in seq_len(harmonics)) {
    cols[[length(cols) + 1]] <- cos(w * h * days)
    cols[[length(cols) + 1]] <- sin(w * h * days)
  }
  p <- df - 1
  cols <- cols[seq_len(p)]
  if (const_col > 0) cols[[const_col]] <- rep(c(0, 5, 1)[(id %% 3) + 1], n)
  x <- do.call(cbind, cols)
  base <- runif(1, 500, 4000)
  y <- base + runif(1, 0, 0.3) * (days - 730000) +
    runif(1, 0, 1500) * cos(w * days) + runif(1, 0, 1500) * sin(w * days) +
    rnorm(n, sd = runif(1, 50, 600))
  y <- pmin(pmax(y, 0), 10000)
  fit <- function(thresh) {
    f <- glmnet(x, y, family = "gaussian", alpha = 1, lambda = 20,
                standardize = TRUE, intercept = TRUE, thresh = thresh)
    as.numeric(coef(f))
  }
  list(id = id, n = n, df = df, lambda = 20,
       x = lapply(seq_len(p), function(j) x[, j]), y = y,
       glmnet_1e4 = fit(1e-4), glmnet_1e12 = fit(1e-12))
}

cases <- list()
ns <- c(12, 14, 16, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150, 200, 24, 36, 45, 70, 90, 180)
dfs <- c(4, 6, 8)
for (i in seq_along(ns)) {
  df <- dfs[(i - 1) %% 3 + 1]
  const_col <- if (i %% 6 == 0) 2 else 0   # a few cases with a constant column
  cases[[i]] <- make_case(i, ns[i], df, const_col)
}
write(toJSON(cases, digits = NA, auto_unbox = TRUE), file.path(outdir, "lasso_glmnet.json"))
