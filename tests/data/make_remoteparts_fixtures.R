## Reference fits of remotePARTS (1.0.4, CRAN) for zeit-core's remotePARTS ports (rust/zeit-core/tests/
## remoteparts.rs): Rscript make_remoteparts_fixtures.R in this folder writes
## ../../rust/zeit-core/tests/data/remoteparts_ar.json, remoteparts_gls.json and remoteparts_parts.json.
##
## ar: remotePARTS::fitAR on simulated series (a linear trend, or a trend and a harmonic, with AR(1)
## errors). fitAR drops the missing values through model.frame and fits the rest as consecutive
## observations: zeit-core's Gaps::Ignore. "exact" is the same REML on the marginal covariance of the
## observations kept (rho^|i - j| between dates i and j), zeit-core's Gaps::Exact; without missing
## values both are fitAR.
##
## gls: remotePARTS::fitGLS on points scattered in the unit square, V = exp(-d / range) of their
## distances, with nugget 0, 0.25 and NA (the maximum-likelihood nugget); designs of a mean, of three
## classes (against a mean: the F test) and of a covariate.
suppressMessages(library(remotePARTS))
suppressMessages(library(jsonlite))
set.seed(20261023)

## fitAR's REML (remotePARTS's AR_fun) on the marginal covariance of the observations at dates idx
exact_fun <- function(par, y, X, idx, logLik.only = TRUE) {
  D <- abs(outer(idx, idx, "-"))
  V <- par^D / (1 - par^2)
  iV <- solve(V)
  n <- length(y)
  q <- ncol(X)
  logdetV <- determinant(V)$modulus[1]
  XtiVX <- t(X) %*% iV %*% X
  beta <- solve(XtiVX, t(X) %*% iV %*% y)
  H <- y - X %*% beta
  s2 <- as.numeric(t(H) %*% iV %*% H) / (n - q)
  ll <- 0.5 * ((n - q) * log(s2) + logdetV + determinant(XtiVX)$modulus[1] + (n - q))
  if (logLik.only) return(ll)
  s2beta <- s2 * solve(XtiVX)
  tstat <- abs(beta[, 1]) / diag(s2beta)^0.5
  list(coefficients = beta[, 1], SE = diag(s2beta)^0.5, tstat = tstat,
       pval = 2 * pt(tstat, df = n - q, lower.tail = FALSE), MSE = s2,
       logLik = 0.5 * (n - q) * log(2 * pi) + determinant(t(X) %*% X)$modulus[1] - ll)
}

fit_exact <- function(y, X, idx) {
  opt <- optim(fn = exact_fun, par = 0.2, y = y, X = X, idx = idx, method = "Brent",
               upper = 1, lower = -1, control = list(maxit = 10^4))
  out <- exact_fun(opt$par, y, X, idx, logLik.only = FALSE)
  out$rho <- opt$par
  out
}

stats <- function(f) {
  list(coefficients = unname(f$coefficients), se = unname(f$SE), abs_t = unname(f$tstat),
       p = unname(f$pval), rho = f$rho, mse = f$MSE, loglik = unname(f$logLik))
}

## AR(1) errors with innovations of sd `sd`, stationary from the first date
ar1 <- function(n, rho, sd) {
  e <- numeric(n)
  e[1] <- rnorm(1, 0, sd / sqrt(1 - rho^2))
  for (i in seq_len(n)[-1]) e[i] <- rho * e[i - 1] + rnorm(1, 0, sd)
  e
}

cases <- list()
add <- function(name, y, harmonic) {
  n <- length(y)
  t <- seq_len(n)
  d <- data.frame(y = y, t = t)
  form <- y ~ t
  if (harmonic) {
    d$s <- sin(2 * pi * t / 12)
    d$c <- cos(2 * pi * t / 12)
    form <- y ~ t + s + c
  }
  X <- model.matrix(update(form, NULL ~ .), d)  # every date's row: y is not in this formula
  ok <- !is.na(y)
  ar <- fitAR(form, data = d)
  ex <- fit_exact(y[ok], X[ok, , drop = FALSE], which(ok))
  if (all(ok)) {
    # without missing values the exact REML is fitAR's (to the optimizer's tolerance where rho
    # goes to the bound, short series)
    stopifnot(isTRUE(all.equal(unname(ex$coefficients), unname(ar$coefficients), tolerance = 1e-4)))
  }
  cases[[length(cases) + 1]] <<- list(
    name = name, y = y, x =lapply(seq_len(ncol(X)), function(j) unname(X[, j])),
    ignore = stats(ar), exact = stats(ex))
}

for (n in c(8, 25, 60, 150)) {
  for (rho in c(-0.6, 0, 0.4, 0.85, 0.97)) {
    for (slope in c(0, 0.004)) {
      y <- 0.7 + slope * seq_len(n) + ar1(n, rho, 0.05)
      add(sprintf("n%d_rho%g_slope%g", n, rho, slope), y, FALSE)
    }
  }
}
for (n in c(36, 120)) {
  for (rho in c(0.2, 0.8)) {
    t <- seq_len(n)
    y <- 0.5 + 0.002 * t + 0.15 * sin(2 * pi * t / 12) + 0.05 * cos(2 * pi * t / 12) + ar1(n, rho, 0.04)
    add(sprintf("harmonic_n%d_rho%g", n, rho), y, TRUE)
  }
}
## missing values: scattered, and a run of them
for (n in c(30, 90)) {
  for (rho in c(0.3, 0.9)) {
    y <- 0.6 - 0.003 * seq_len(n) + ar1(n, rho, 0.05)
    y[sample.int(n, round(0.2 * n))] <- NA
    add(sprintf("gaps_n%d_rho%g", n, rho), y, FALSE)
    y <- 0.6 - 0.003 * seq_len(n) + ar1(n, rho, 0.05)
    y[seq(n %/% 3, n %/% 3 + 6)] <- NA
    add(sprintf("run_n%d_rho%g", n, rho), y, FALSE)
  }
}
## a large offset (a temperature in kelvin): the fit must not lose it in sums of squares
y <- 300 + 0.01 * seq_len(40) + ar1(40, 0.5, 0.1)
add("kelvin_n40", y, FALSE)

out_dir <- file.path("..", "..", "rust", "zeit-core", "tests", "data")
write(toJSON(list(remoteparts = as.character(packageVersion("remotePARTS")), ar = cases),
             digits = NA, auto_unbox = TRUE, na = "null"),
      file.path(out_dir, "remoteparts_ar.json"))
cat("ar:", length(cases), "cases\n")

gls <- list()
for (n in c(150, 400)) {
  for (range in c(0.1, 0.3)) {
    coords <- cbind(runif(n), runif(n))
    V <- exp(-as.matrix(dist(coords)) / range)
    # the data's covariance: V with a nugget of 0.2
    U <- chol(0.8 * V + 0.2 * diag(n))
    d <- data.frame(class = factor(sample(c("a", "b", "c"), n, replace = TRUE)), cov = rnorm(n))
    e <- as.numeric(t(U) %*% rnorm(n)) * 0.01
    d$y <- 0.002 + c(a = 0, b = 0.004, c = -0.002)[as.character(d$class)] + 0.003 * d$cov + e
    for (design in c("mean", "classes", "covariate")) {
      form <- switch(design, mean = y ~ 1, classes = y ~ 0 + class, covariate = y ~ cov)
      X <- model.matrix(form, d)
      for (nugget in c(0, 0.25, NA)) {
        # y ~ 1 is its own null model: fitGLS warns that there is no F test
        f <- suppressWarnings(fitGLS(form, data = d, V = V, nugget = nugget))
        r <- list(coefficients = unname(f$coefficients), se = unname(f$SE), t = unname(f$tstat),
                  p = unname(f$pval_t), sse = as.numeric(f$SSE), mse = as.numeric(f$MSE),
                  logdet_v = f$logDetV, loglik = f$logLik, nugget = f$nugget)
        if (design != "mean") {
          r$null <- list(coefficients = as.numeric(f$coefficients0), se = as.numeric(f$SE0),
                         sse = as.numeric(f$SSE0), mse = f$MSE0, loglik = f$logLik0, msr = f$MSR,
                         f = as.numeric(f$Fstat), df = as.numeric(f$df_F), p = as.numeric(f$pval_F))
        }
        gls[[length(gls) + 1]] <- list(
          name = sprintf("n%d_range%g_%s_nugget%s", n, range, design, ifelse(is.na(nugget), "ML", nugget)),
          x_coord = coords[, 1], y_coord = coords[, 2], range = range, nugget = nugget, y = d$y,
          x = lapply(seq_len(ncol(X)), function(j) unname(X[, j])), fit = r)
      }
    }
  }
}
write(toJSON(list(remoteparts = as.character(packageVersion("remotePARTS")), gls = gls),
             digits = NA, auto_unbox = TRUE, na = "null"),
      file.path(out_dir, "remoteparts_gls.json"))
cat("gls:", length(gls), "cases\n")

## parts: fitCor and fitGLS_partition on a map of 2000 pixels in the unit square, with Euclidean
## distances divided by their largest (distm_scaled's scaling, without geosphere's geodesics)
distm_euc <- function(coords, coords2 = NULL) {
  coords <- as.matrix(coords)
  if (is.null(coords2)) {
    D <- as.matrix(dist(coords))
  } else {
    coords2 <- as.matrix(coords2)
    D <- sqrt(outer(coords[, 1], coords2[, 1], "-")^2 + outer(coords[, 2], coords2[, 2], "-")^2)
  }
  D / max(D)
}
cm <- function(m) list(dim = dim(m), values = as.numeric(m))  # column-major
npix <- 2000
map <- data.frame(lng = runif(npix), lat = runif(npix))
U <- chol(0.9 * exp(-distm_euc(map) / 0.08) + 0.1 * diag(npix))
field <- function() as.numeric(t(U) %*% rnorm(npix))
## residual series: AR(1) in time of the spatially correlated innovations, of the first 300 pixels
nt <- 30
res <- matrix(0, npix, nt)
res[, 1] <- field()
for (t in 2:nt) res[, t] <- 0.4 * res[, t - 1] + field()
sub <- 1:300
cor_exp <- fitCor(res, map, distm_FUN = "distm_euc", covar_FUN = "covar_exp", start = list(range = 0.1),
                  index = sub)
cor_pow <- fitCor(res, map, distm_FUN = "distm_euc", covar_FUN = "covar_exppow",
                  start = list(range = 0.1, shape = 1), index = sub)
fitcor <- list(resids = cm(res[sub, ]), exponential = unname(cor_exp$spcor),
               exponential_power = unname(cor_pow$spcor), max_distance = cor_exp$max.distance)

map$class <- factor(sample(c("a", "b", "c"), npix, replace = TRUE))
map$cov <- rnorm(npix)
map$y <- 0.002 + c(a = 0, b = 0.001, c = -0.0005)[as.character(map$class)] + 0.0004 * map$cov +
  0.001 * field()
pm <- sample_partitions(npix = npix, partsize = 200, npart = 10)
part_case <- function(name, form, nugget, chisqr) {
  # progressbar = FALSE fails in 1.0.4 ('pb' not found): the bar is drawn and dropped
  invisible(capture.output(f <- suppressWarnings(fitGLS_partition(
    form, partmat = pm, data = map, covar_FUN = "covar_exp", covar.pars = c(range = 0.1),
    distm_FUN = "distm_euc", nugget = nugget, ncross = 6, do.chisqr.test = chisqr))))
  X <- model.matrix(form, map)
  list(name = name, nugget = nugget, x = lapply(seq_len(ncol(X)), function(j) unname(X[, j])),
       coefficients = cm(f$part$coefficients), se = cm(f$part$SEs), nuggets = f$part$nuggets,
       f = unname(f$part$modstats[, "Fstats"]), rcoefs = cm(f$cross$rcoefs), rssrs = f$cross$rSSRs,
       rsses = f$cross$rSSEs, overall = list(
         coefficients = unname(f$overall$coefficients), rcoefficients = cm(f$overall$rcoefficients),
         rssr = f$overall$rSSR, rsse = f$overall$rSSE, f = f$overall$Fstat, dfs = unname(f$overall$dfs),
         t = cm(unname(f$overall$t.test)), covar = cm(unname(f$overall$covar_coef)),
         p_chisqr = if (chisqr) as.numeric(f$overall$pval.chisqr) else NULL))
}
parts <- list(
  part_case("classes_nugget0", y ~ 0 + class, 0, TRUE),
  part_case("mean_nugget0", y ~ 1, 0, FALSE),
  part_case("covariate_nuggetML", y ~ cov, NA, TRUE))
chisqr_cases <- lapply(list(c(1.3, 0.004, 2, 10), c(2.5, 0.02, 1, 30), c(0.8, 0.0, 3, 5),
                            c(1.1, 0.4, 2, 8)), function(a) {
  list(fmean = a[1], rssr = a[2], df1 = a[3], npart = a[4],
       p = as.numeric(remotePARTS:::part_chisqr(a[1], a[2], a[3], a[4])))
})
write(toJSON(list(remoteparts = as.character(packageVersion("remotePARTS")), lng = map$lng, lat = map$lat,
                  y = map$y, partitions = as.integer(pm) - 1L, partsize = nrow(pm), fitcor = fitcor,
                  parts = parts, chisqr = chisqr_cases),
             digits = NA, auto_unbox = TRUE, na = "null"),
      file.path(out_dir, "remoteparts_parts.json"))
cat("parts:", length(parts), "cases\n")
