# Reference of zeit.train_di and zeit.aoa: CAST's trainDI and aoa (CAST 1.0.4, GPL >= 2) on seeded data: training
# points in clusters, with weights and cross-validation folds or without, and new points around and far from them.
#
#   Rscript tests/data/make_aoa_cast_fixtures.R
#
# writes tests/data/aoa_cast.json next to this script.
suppressMessages(library(CAST))
library(jsonlite)

make_case <- function(seed, n, p, k, weighted, folds) {
  set.seed(seed)
  centres <- matrix(rnorm(k * p, sd = 4), k, p)
  group <- sample(seq_len(k), n, replace = TRUE)
  train <- centres[group, , drop = FALSE] + matrix(rnorm(n * p), n, p)
  train[, 1] <- train[, 1] * 100 + 500        # predictors in different units
  colnames(train) <- paste0("v", seq_len(p))
  train <- as.data.frame(train)
  m <- 300
  newdata <- matrix(rnorm(m * p, sd = 6), m, p)
  newdata[, 1] <- newdata[, 1] * 100 + 500
  newdata[1:20, ] <- as.matrix(train[1:20, ]) + matrix(rnorm(20 * p, sd = 0.05), 20, p)
  newdata[21, 2] <- NA
  colnames(newdata) <- colnames(train)
  newdata <- as.data.frame(newdata)
  weight <- if (weighted) round(runif(p, 0.2, 3), 3) else rep(1, p)
  wdf <- as.data.frame(t(weight))
  names(wdf) <- colnames(train)
  cv <- if (folds) group else NULL
  tdi <- suppressMessages(trainDI(train = train, weight = wdf, CVtest = cv, verbose = FALSE))
  a <- suppressMessages(aoa(newdata, train = train, trainDI = tdi, LPD = TRUE, verbose = FALSE))
  list(
    train = unname(as.matrix(train)), weight = weight, folds = if (folds) group else NULL,
    newdata = unname(as.matrix(newdata)),
    train_di = tdi$trainDI, threshold = unname(tdi$threshold), mean_distance = tdi$trainDist_avrgmean,
    di = a$DI, aoa = a$AOA, lpd = a$LPD
  )
}

cases <- list(
  make_case(1, 80, 4, 5, weighted = FALSE, folds = FALSE),
  make_case(2, 120, 6, 6, weighted = TRUE, folds = TRUE),
  make_case(3, 60, 3, 4, weighted = TRUE, folds = FALSE)
)

path <- file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "aoa_cast.json")
writeLines(toJSON(list(cast = as.character(packageVersion("CAST")), cases = cases),
                  digits = NA, na = "null", matrix = "rowmajor", auto_unbox = TRUE, null = "null"), path)
cat("wrote", path, "\n")
