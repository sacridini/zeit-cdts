# Reference of zeit.spatial_folds(method="knndm"): CAST's knndm (CAST 1.0.4, GPL >= 2) on its documented examples,
# 500 training points in a 100 x 100 square, clustered (25 parents, radius 5) or at random, and 1000 regular
# prediction points: the points, CAST's folds and its W statistic.
#
#   Rscript tests/data/make_knndm_cast_fixtures.R
#
# writes tests/data/knndm_cast.json next to this script.
suppressMessages({library(CAST); library(sf)})
library(jsonlite)

area <- st_polygon(list(matrix(c(0, 0, 0, 100, 100, 100, 100, 0, 0, 0), ncol = 2, byrow = TRUE)))
make_case <- function(clustered) {
  set.seed(1234)
  tp <- if (clustered) clustered_sample(area, 500, 25, 5) else st_sample(area, 500, type = "random")
  pp <- st_sample(area, 1000, type = "regular")
  kf <- suppressWarnings(knndm(tp, predpoints = pp, k = 5))
  list(clustered = clustered, train = unname(st_coordinates(tp)[, 1:2]), pred = unname(st_coordinates(pp)[, 1:2]),
       folds = kf$clusters - 1, W = kf$W)
}
path <- file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "knndm_cast.json")
writeLines(toJSON(list(cast = as.character(packageVersion("CAST")), cases = list(make_case(TRUE), make_case(FALSE))),
                  digits = NA, matrix = "rowmajor", auto_unbox = TRUE), path)
cat("wrote", path, "\n")
