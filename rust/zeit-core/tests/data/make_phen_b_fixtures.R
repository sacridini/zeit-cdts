# Fixtures of tests/phen_b_ref.rs: the curve fits of the phenofit R package
# (0.3.11), through its documented FitDL.Beck/Elmore/Gu/Zhang/AG/Klos, and its
# documented extraction functions PhenoTrs (approach "White"), PhenoDeriv and
# PhenoGu, on synthetic series generated here (46 points, t = 1, 9, ..., 361).
# phenofit's Klosterman and Zhang curvature dates could not be obtained from
# its documented interface (NA), so those metrics are not compared. Zeit's
# optimiser, parametrisations, bounds and metric definitions are not
# phenofit's: the test measures the agreement rather than demanding equality.
#
#   Rscript make_phen_b_fixtures.R   (writes phen_b_ref.json next to this file)

suppressMessages({
  library(phenofit)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = FALSE)
here <- dirname(sub("^--file=", "", args[grep("^--file=", args)]))
if (length(here) == 0 || here == "") here <- "."

set.seed(20261106)
t <- seq(1, 365, by = 8)
tout <- 1:365
sig <- function(x) 1 / (1 + exp(-x))

make_series <- function(k) {
  mn <- runif(1, 0.05, 0.2)
  amp <- runif(1, 0.4, 0.7)
  sos <- runif(1, 40, 110)
  eos <- runif(1, 210, 300)
  rsp <- runif(1, 0.05, 0.15)
  rau <- runif(1, 0.05, 0.15)
  m7 <- if (k > 16) runif(1, 0.0001, 0.0006) else 0
  sigma <- sample(c(0.01, 0.02, 0.03), 1)
  y <- mn + (amp - m7 * t) * (sig(rsp * (t - sos)) - sig(rau * (t - eos))) + rnorm(length(t), 0, sigma)
  list(y = y, truth = c(mn = mn, amp = amp, sos = sos, rsp = rsp, eos = eos, rau = rau, m7 = m7), sigma = sigma)
}

rmse <- function(y, f) sqrt(mean((y - f)^2))
nse <- function(y, f) 1 - sum((y - f)^2) / sum((y - mean(y))^2)

metrics_of <- function(fit) {
  d1 <- c(diff(fit), 0)
  num <- function(x, nm) {
    v <- suppressWarnings(as.numeric(x[nm]))
    if (length(v) == 0 || is.na(v)) NA_real_ else v
  }
  out <- c()
  for (tr in c(0.2, 0.5, 0.6)) {
    p <- tryCatch(suppressWarnings(PhenoTrs(fit, tout, approach = "White", trs = tr, IsPlot = FALSE)), error = function(e) NULL)
    out <- c(out, num(p, "sos"), num(p, "eos"))
  }
  p <- tryCatch(suppressWarnings(PhenoDeriv(fit, tout, der1 = d1, IsPlot = FALSE)), error = function(e) NULL)
  out <- c(out, num(p, "sos"), num(p, "pos"), num(p, "eos"))
  p <- tryCatch(suppressWarnings(PhenoGu(fit, tout, der1 = d1, IsPlot = FALSE)), error = function(e) NULL)
  out <- c(out, num(p, "UD"), num(p, "SD"), num(p, "DD"), num(p, "RD"))
  out
}

curves <- list(beck = FitDL.Beck, elmore = FitDL.Elmore, gu = FitDL.Gu, zhang = FitDL.Zhang, ag = FitDL.AG, klos = FitDL.Klos)

cases <- list()
for (k in 1:24) {
  s <- make_series(k)
  fits <- list()
  for (nm in names(curves)) {
    r <- tryCatch(suppressWarnings(curves[[nm]](s$y, t, tout = tout)), error = function(e) NULL)
    if (is.null(r)) next
    fit <- as.numeric(r$zs$iter2)
    fo <- fit[match(t, tout)]
    fits[[nm]] <- list(
      par = as.numeric(r$par[1, ]),
      par_names = colnames(r$par),
      curve = round(fit, 6),
      rmse = rmse(s$y, fo),
      nse = nse(s$y, fo),
      metrics = metrics_of(fit)
    )
  }
  cases[[k]] <- list(y = round(s$y, 6), truth = as.list(s$truth), sigma = s$sigma, fits = fits)
}

out <- list(t = t, tout = tout, cases = cases)
write(toJSON(out, auto_unbox = TRUE, digits = 8, na = "null"), file.path(here, "phen_b_ref.json"))
cat("wrote", file.path(here, "phen_b_ref.json"), "\n")
