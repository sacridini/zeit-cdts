# Reference of zeit.emerging_hotspots: sfdep's emerging_hotspot_analysis (sfdep 0.2.5, GPL-3, run as a black box:
# its outputs only) on seeded grids of cells (queen neighbours), with time lags k = 1 and 2: the Gi* of every cell
# at every date (include_gi = TRUE) and the Mann-Kendall tau and p of each cell's Gi* (Kendall::MannKendall).
#
#   Rscript tests/data/make_hotspots_sfdep_fixtures.R
#
# writes tests/data/hotspots_sfdep.json next to this script: values and gi as (time, row, column), the rows from the
# grid's bottom (st_make_grid's order), tau and p as (row, column).
suppressMessages({library(sfdep); library(sf); library(jsonlite)})

make_case <- function(seed, nx, ny, nt, k) {
  set.seed(seed)
  g <- st_make_grid(st_bbox(c(xmin = 0, ymin = 0, xmax = nx, ymax = ny)), n = c(nx, ny))
  geo <- st_sf(id = sprintf("%04d", seq_along(g)), geometry = g)
  df <- expand.grid(id = geo$id, t = seq_len(nt), stringsAsFactors = FALSE)
  i <- as.integer(df$id) - 1; x <- i %% nx; y <- i %/% nx
  bump <- function(cx, cy, r) exp(-((x - cx)^2 + (y - cy)^2) / (2 * r^2))
  df$value <- rnorm(nrow(df)) + 3 * bump(3, 3, 1.5) * df$t / nt + 3 * bump(8, 6, 1.5) * (1 - df$t / nt) -
    2 * bump(9, 1, 1.2) * (df$t > nt - 3)
  df$time <- as.Date("2000-01-01") + df$t * 365
  st <- spacetime(df, geo, .loc_col = "id", .time_col = "time")
  e <- emerging_hotspot_analysis(st, "value", k = k, nsim = 9, include_gi = TRUE)
  gi <- attr(e, "gi_star")$gi_star
  out <- as.data.frame(st)
  cube <- function(v) aperm(array(v[order(out$t, out$id)], c(nx, ny, nt)), c(3, 2, 1))
  loc <- function(v) t(matrix(v[order(e$location)], nx, ny))
  list(k = k, values = cube(out$value), gi = cube(gi), tau = loc(e$tau), p = loc(e$p_value))
}

cases <- list(make_case(1, 12, 10, 10, 1), make_case(2, 9, 11, 14, 2))
path <- file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "hotspots_sfdep.json")
writeLines(toJSON(list(sfdep = as.character(packageVersion("sfdep")), cases = cases),
                  digits = NA, auto_unbox = TRUE), path)
cat("wrote", path, "\n")
