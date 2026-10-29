# Reference of zeit.error_profile: CAST's errorProfiles (CAST 1.0.4, GPL >= 2; scam 1.2.23) on seeded data, a
# regression (caret's lm, RMSE) and a classification (caret's rpart, accuracy), cross-validated in 5 folds: the
# predictors, the predictions out of fold, the folds, CAST's weights (the model's variable importance), its
# threshold, the moving windows it fits (DI, metric) and its curve on a grid of DI.
#
#   Rscript tests/data/make_error_profile_cast_fixtures.R
#
# writes tests/data/error_profile_cast.json next to this script.
suppressMessages({library(CAST); library(caret)})
library(jsonlite)

make_case <- function(classes) {
  set.seed(3)
  n <- 150
  x <- data.frame(a = c(rnorm(100), rnorm(50, 3)), b = rnorm(n), c = runif(n))
  y <- 2 * x$a + x$b^2 + rnorm(n, sd = 0.5 + abs(x$a))
  if (classes) y <- factor(ifelse(y > median(y), "high", "low"))
  folds <- sample(rep(1:5, length.out = n))
  idx <- lapply(1:5, function(f) which(folds != f))
  ctrl <- trainControl(method = "cv", index = idx, savePredictions = "final")
  m <- train(x, y, method = if (classes) "rpart" else "lm", trControl = ctrl)
  td <- trainDI(m, verbose = FALSE)
  ep <- errorProfiles(m, td)
  pred <- m$pred[order(m$pred$rowIndex), ]
  grid <- seq(min(ep$model$DI), max(ep$model$DI), length.out = 50)
  list(classes = classes, x = unname(as.matrix(x)), observed = as.character(pred$obs),
       predicted = as.character(pred$pred), folds = folds, weight = unname(unlist(td$weight)),
       threshold = td$threshold, window_di = ep$model$DI, window_metric = ep$model$metric,
       grid = grid, curve = unname(predict(ep, data.frame(DI = grid))))
}
path <- file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))),
                  "error_profile_cast.json")
writeLines(toJSON(list(cast = as.character(packageVersion("CAST")), scam = as.character(packageVersion("scam")),
                       cases = list(make_case(FALSE), make_case(TRUE))),
                  digits = NA, matrix = "rowmajor", auto_unbox = TRUE), path)
cat("wrote", path, "\n")
