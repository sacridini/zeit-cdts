# Reference of zeit.beast: Rbeast's beast() and beast.irreg() (Rbeast 1.0.2, GPL >= 2), with the generic
# kernels (cputype = 'sse') and fixed seeds, on the Nile's flow (trend only), Rbeast's simulated series and its
# Yellowstone NDVI (season and trend, with gaps), the irregular Landsat NDVI of its Ohio site, a short series and
# other priors. zeit.beast is a port of the same C code, so the outputs agree to the last bit of their float32.
#
#   Rscript tests/data/make_beast_fixtures.R
#
# writes tests/data/beast_rbeast.json next to this script.
suppressMessages(library(Rbeast))
library(jsonlite)

out_component <- function(c, trend) {
  r <- list(ncp = c$ncp, ncp_mode = c$ncp_mode, ncp_median = c$ncp_median, ncp_pct90 = c$ncp_pct90,
            ncp_pct10 = c$ncp_pct10, ncp_prob = c$ncpPr, cp_occ = c$cpOccPr, order = c$order, y = c$Y, sd = c$SD,
            ci = if (is.null(c$CI)) NULL else unname(as.matrix(c$CI)),
            cp = c$cp, cp_prob = c$cpPr, cp_abrupt = c$cpAbruptChange, cp_ci = unname(as.matrix(c$cpCI)))
  if (trend) {
    r$slope <- c$slp
    r$slope_sd <- c$slpSD
    r$slope_pos_prob <- c$slpSgnPosPr
    r$slope_zero_prob <- c$slpSgnZeroPr
    r$slope_ci <- if (is.null(c$slpCI)) NULL else unname(as.matrix(c$slpCI))
  } else {
    r$amp <- c$amp
    r$amp_sd <- c$ampSD
  }
  r
}

# args: the zeit.beast arguments; the same are given to Rbeast under its names.
make_case <- function(name, y, args, time = NULL) {
  season <- if (is.null(args$season)) "harmonic" else args$season
  r_args <- list(season = season, mcmc.seed = args$seed, mcmc.chains = args$chains,
                 mcmc.samples = if (is.null(args$samples)) 8000 else args$samples,
                 dump.ci = if (is.null(args$ci)) TRUE else args$ci,
                 precPriorType = if (is.null(args$prec_type)) "componentwise" else args$prec_type,
                 quiet = TRUE, print.progress = FALSE, print.param = FALSE, cputype = "sse")
  if (!is.null(args$period)) r_args$period <- args$period
  if (!is.null(args$deltat)) r_args$deltat <- args$deltat
  if (!is.null(args$tcp_minmax)) r_args$tcp.minmax <- args$tcp_minmax
  if (!is.null(args$torder_minmax)) r_args$torder.minmax <- args$torder_minmax
  if (!is.null(args$scp_minmax)) r_args$scp.minmax <- args$scp_minmax
  if (is.null(time)) {
    if (!is.null(args$start)) r_args$start <- args$start
    o <- do.call(beast, c(list(y), r_args))
  } else {
    o <- do.call(beast.irreg, c(list(y, time = time), r_args))
  }
  list(name = name, args = args, y = y, times = time,
       time = o$time, data = o$data, marg_lik = o$marg_lik, sig2 = o$sig2, r2 = o$R2, rmse = o$RMSE,
       trend = out_component(o$trend, TRUE),
       season = if (is.null(o$season)) NULL else out_component(o$season, FALSE))
}

data(Yellowstone)
data(simdata)
data(ohio)
yell <- as.numeric(Yellowstone)
yell[seq(5, length(yell), by = 9)] <- NA       # gaps: every 9th value and a year and a half
yell[200:235] <- NA

cases <- list(
  make_case("nile", as.numeric(Nile), list(season = "none", start = 1871, deltat = 1, chains = 3, seed = 1)),
  make_case("simdata", simdata[, 1], list(period = 24, chains = 2, seed = 11, samples = 2000)),
  make_case("yellowstone_gaps", yell, list(period = 1, deltat = 1 / 24, start = 1981.5, chains = 1, seed = 3,
                                           samples = 3000, prec_type = "uniform")),
  make_case("ohio_irregular", ohio$ndvi, list(period = 1, deltat = 1 / 12, chains = 2, seed = 42, samples = 2000),
            time = ohio$time),
  make_case("short", c(1, 2, 1, 3, 2, 8, 9, 8, 10, 9, 9, 8, 2, 3, 2, 1, 2, 3, 2, 1),
            list(season = "none", chains = 3, seed = 5, tcp_minmax = c(0, 3))),
  make_case("nile_priors", as.numeric(Nile), list(season = "none", start = 1871, deltat = 1, chains = 2, seed = 9,
                                                  ci = FALSE, prec_type = "constant", torder_minmax = c(0, 2)))
)

path <- file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "beast_rbeast.json")
writeLines(toJSON(list(rbeast = as.character(packageVersion("Rbeast")), cases = cases),
                  digits = NA, na = "null", matrix = "rowmajor", auto_unbox = TRUE, null = "null"), path)
cat("wrote", path, "\n")
