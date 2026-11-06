# Fixtures of tests/phen_a_ref.rs: the smoothing and the season split of the
# phenofit R package (0.3.11), run through its documented functions only
# (check_input, smooth_wWHIT, wTSM, season_mov), on synthetic series generated
# here. Zeit's rules are not phenofit's, so the test measures the agreement
# rather than demanding equality.
#
#   Rscript make_phen_a_fixtures.R   (writes phen_a_ref.json next to this file)

suppressMessages({
  library(phenofit)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = FALSE)
here <- dirname(sub("^--file=", "", args[grep("^--file=", args)]))
if (length(here) == 0 || here == "") here <- "."

set.seed(42)
npt <- 23
n <- 69
t <- as.Date("2000-01-01") + 16 * (0:(n - 1))
d <- as.numeric(t - t[1])

make_series <- function(kind) {
  amp <- runif(1, 0.2, 0.35)
  sigma <- runif(1, 0.01, 0.03)
  freq <- if (kind == 1) 1 else 2
  truth <- 0.5 + amp * sin(2 * pi * freq * (d - 80) / 365.25)
  y <- truth + rnorm(n, 0, sigma)
  frac <- runif(1, 0.05, 0.30)
  cl <- runif(n) < frac
  y[cl] <- y[cl] - runif(sum(cl), 0.1, 0.4)
  list(y = y, truth = truth)
}

nearest <- function(date) which.min(abs(as.numeric(t - date)))

cases <- list()
for (k in 1:200) {
  kind <- if (k <= 120) 1 else 2
  s <- make_series(kind)
  y <- s$y
  w <- rep(1, n)
  res <- tryCatch({
    INPUT <- check_input(t, y, w, nptperyear = npt, maxgap = 11, alpha = 0.02, wmin = 0.2)
    sm <- smooth_wWHIT(INPUT$y, INPUT$w, INPUT$ylu, npt, wFUN = wTSM, iters = 3, lambda = 15)
    brks <- season_mov(INPUT, list(rFUN = "smooth_wWHIT", wFUN = "wTSM", lambda = 15,
                                   iters = 3, wmin = 0.1, calendarYear = FALSE))
    w_ph <- wTSM(INPUT$y, sm$zs$ziter1, INPUT$w, iter = 2, nptperyear = npt, wfact = 0.5)
    dt <- brks$dt
    list(
      kind = kind,
      y = round(y, 6),
      truth = round(s$truth, 6),
      # the series phenofit worked on (check_input may change y)
      y_input = round(INPUT$y, 6),
      w_input = round(INPUT$w, 6),
      z = round(sm$zs$ziter3, 6),
      z_one = round(sm$zs$ziter1, 6),
      w_tsm = round(w_ph, 6),
      beg = vapply(dt$beg, nearest, 1) - 1,
      peak = vapply(dt$peak, nearest, 1) - 1,
      end = vapply(dt$end, nearest, 1) - 1
    )
  }, error = function(e) NULL)
  if (!is.null(res)) cases[[length(cases) + 1]] <- res
}

out <- list(nptperyear = npt, lambda = 15, t = d, cases = cases)
write_json(out, file.path(here, "phen_a_ref.json"), auto_unbox = TRUE, digits = 6)
cat("series:", length(cases), "\n")
