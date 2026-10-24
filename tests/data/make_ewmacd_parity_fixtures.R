## Reference flags of Brooks' EWMACD v1.3.0 script (Brooks et al. 2014's) for tests/test_ewmacd.py:
## Rscript make_ewmacd_parity_fixtures.R "EWMACD v1.3.0.r" in this folder writes ewmacd_brooks_parity.json.
## The script is in "EWMACD v1.3.0.7z" at https://vtechworks.lib.vt.edu/items/version/7 (no licence is
## stated: it is run here, never copied into zeit). Only its function definitions are evaluated, so
## the packages it installs and loads (raster, snow...) are not needed.
suppressMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
for (e in parse(file = args[1])) {
  if (is.call(e) && (identical(e[[1]], as.name("<-")) || identical(e[[1]], as.name("="))) &&
      is.call(e[[3]]) && identical(e[[3]][[1]], as.name("function"))) {
    eval(e, envir = globalenv())
  }
}

## irregular dates 2005-2012, seasonal pixels with noise, clouds and changes (no missing training date:
## the script's refit misreads the values when there are)
set.seed(2014)
years <- c(); doys <- c()
for (y in 2005:2012) {
  d <- sample(1:20, 1)
  while (d <= 365) {
    if (runif(1) < 0.6) { years <- c(years, y); doys <- c(doys, d) }
    d <- d + sample(10:40, 1)
  }
}
t <- years + doys / 365
pixels <- list()
for (k in 0:17) {
  base <- runif(1, 300, 700); a1 <- runif(1, -150, 150); b1 <- runif(1, -150, 150)
  a2 <- runif(1, -50, 50); b2 <- runif(1, -50, 50); noise <- runif(1, 5, 40)
  kind <- k %% 6; t0 <- 2009 + runif(1, 0, 2.5)
  th <- 2 * pi * doys / 365
  v <- base + a1 * sin(th) + b1 * cos(th) + a2 * sin(2 * th) + b2 * cos(2 * th) + rnorm(length(t), 0, noise)
  if (kind == 1) v <- v - ifelse(t >= t0, 300 * pmax(0, 1 - (t - t0) / 3), 0)
  if (kind == 2) v <- v - ifelse(t >= t0, 3 * noise, 0)
  if (kind == 3) v <- v + ifelse(t >= t0, 4 * noise, 0)
  if (kind == 4) v <- v - ifelse(t < 2007, 200, 0)
  cloud <- runif(length(t)) < 0.06
  v[cloud] <- v[cloud] - runif(sum(cloud), 200, 500)
  if (kind == 5) v[years >= 2009 & runif(length(t)) < 0.08] <- NA
  pixels[[k + 1]] <- round(v)
}
hb <- max(which(years < 2009))
cases <- list()
for (cfg in list(c(0.3, 2), c(0.15, 1), c(0.5, 3))) {
  run <- function(rounding) lapply(pixels, function(v) EWMACD.pixel.for.rasterEngine(
    myPixel = v, nc = cfg[2], ns = cfg[2], DOYs = doys, Years = years, historybound = hb, xBarLimit1 = 1.5,
    xBarLimit2 = 20, lowthresh = -1e9, lambdasigs = 3, lambda = cfg[1], rounding = rounding, persistence = 1,
    trainingStart = 2005, trainingEnd = 2009, testingEnd = 2013))
  cases[[length(cases) + 1]] <- list(lmbda = cfg[1], harmonics = cfg[2], flags = run(TRUE), ewma = run(FALSE))
}
out <- list(source = "EWMACD v1.3.0 (Brooks et al. 2014), persistence 1, lowthresh -1e9, xBarLimits 1.5 and 20, 3 sigmas",
            train_start = 2005, train_end = 2009, years = years, doys = doys,
            values = lapply(pixels, function(v) ifelse(is.na(v), NA, v)), cases = cases)
cat(toJSON(out, digits = NA, auto_unbox = TRUE, na = "null"), file = "ewmacd_brooks_parity.json")
