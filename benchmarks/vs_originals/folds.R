# The originals of zeit's spatial cross-validation (benchmarks/vs_originals/folds.py), each call timed:
#   autocor <raster>                       blockCV's cv_spatial_autocor: the range of each band (<raster>_range.txt)
#   knndm <n> <out prefix>                 CAST's knndm on n clustered points (its example) and 1000 regular
#                                          prediction points: the points, folds and W written next to the prefix
#   profile <x.csv> <y.txt> <folds.txt>    CAST's errorProfiles of a cross-validated lm (caret): its windows and
#                                          weights written next to x.csv
#   md <train.csv> <new.csv>               CAST's trainDI and aoa with method = "MD": the DI next to new.csv
args <- commandArgs(TRUE)
mode <- args[1]
suppressMessages({library(CAST); library(sf)})

if (mode == "autocor") {
  suppressMessages({library(blockCV); library(terra)})
  r <- rast(args[2])
  set.seed(1)
  s <- system.time(a <- suppressMessages(cv_spatial_autocor(r, num_sample = 5000, plot = FALSE,
                                                             progress = FALSE)))[["elapsed"]]
  writeLines(format(a$range_table$range, digits = 17), sub(".tif", "_range.txt", args[2], fixed = TRUE))
  cat("SECONDS", s, "\n")
} else if (mode == "knndm") {
  n <- as.integer(args[2])
  area <- st_polygon(list(matrix(c(0, 0, 0, 100, 100, 100, 100, 0, 0, 0), ncol = 2, byrow = TRUE)))
  set.seed(1234)
  tp <- clustered_sample(area, n, n / 20, 5)
  pp <- st_sample(area, 1000, type = "regular")
  s <- system.time(kf <- suppressWarnings(knndm(tp, predpoints = pp, k = 5)))[["elapsed"]]
  write.csv(st_coordinates(tp)[, 1:2], paste0(args[3], "_train.csv"), row.names = FALSE)
  write.csv(st_coordinates(pp)[, 1:2], paste0(args[3], "_pred.csv"), row.names = FALSE)
  writeLines(as.character(kf$clusters - 1), paste0(args[3], "_folds.txt"))
  writeLines(format(kf$W, digits = 17), paste0(args[3], "_W.txt"))
  cat("SECONDS", s, "\n")
} else if (mode == "profile") {
  suppressMessages(library(caret))
  x <- read.csv(args[2]); y <- scan(args[3], quiet = TRUE); folds <- scan(args[4], quiet = TRUE)
  idx <- lapply(sort(unique(folds)), function(f) which(folds != f))
  m <- train(x, y, method = "lm", trControl = trainControl(method = "cv", index = idx, savePredictions = "final"))
  pred <- m$pred[order(m$pred$rowIndex), ]
  s <- system.time({
    td <- trainDI(m, verbose = FALSE)
    ep <- errorProfiles(m, td)
  })[["elapsed"]]
  base <- sub(".csv", "", args[2], fixed = TRUE)
  writeLines(format(pred$pred, digits = 17), paste0(base, "_pred.txt"))
  writeLines(format(unlist(td$weight), digits = 17), paste0(base, "_weight.txt"))
  write.csv(data.frame(DI = ep$model$DI, metric = ep$model$metric), paste0(base, "_windows.csv"), row.names = FALSE)
  grid <- seq(min(ep$model$DI), max(ep$model$DI), length.out = 200)
  write.csv(data.frame(DI = grid, fit = predict(ep, data.frame(DI = grid))), paste0(base, "_curve.csv"),
            row.names = FALSE)
  cat("SECONDS", s, "\n")
} else if (mode == "md") {
  train <- read.csv(args[2], header = FALSE); newdata <- read.csv(args[3], header = FALSE)
  names(newdata) <- names(train)
  s <- system.time({
    tdi <- suppressMessages(trainDI(train = train, method = "MD", verbose = FALSE))
    a <- suppressMessages(aoa(newdata, train = train, trainDI = tdi, method = "MD", verbose = FALSE))
  })[["elapsed"]]
  writeLines(format(a$DI, digits = 17), sub(".csv", "_di.txt", args[3], fixed = TRUE))
  cat("SECONDS", s, "\n")
}
