# The original of zeit.transitions (benchmarks/vs_originals/accuracy_survey.py): the area of each transition between
# two maps of classes inside the strata (terra's freq of the pairs) and its estimate from a stratified sample by
# R's survey package (Lumley): svydesign(ids = ~1, strata = ~h, weights = ~A_h / n_h), svytotal (direct), the mapped
# area plus svytotal of y - x (difference), svyratio(~y, ~x) times the mapped area (ratio), svyratio for the user's
# and producer's accuracies, svymean for the overall; timed together.
#   Rscript accuracy_survey.R <folder> <estimator>     strata.tif, map_from.tif, map_to.tif, points.csv (row, col,
#                                                      ref_from, ref_to) in the folder; survey_<estimator>.json
args <- commandArgs(TRUE)
d <- args[1]
estimator <- args[2]
suppressMessages({library(survey); library(terra); library(jsonlite)})
terraOptions(progress = 0)
pts <- read.csv(file.path(d, "points.csv"))

s <- system.time({
    st <- rast(file.path(d, "strata.tif"))
    pair <- mask(rast(file.path(d, "map_from.tif")) * 100 + rast(file.path(d, "map_to.tif")), st)
    px <- prod(res(st)) / 1e4  # ha
    f <- freq(pair)
    X <- setNames(f$count * px, paste(f$value %/% 100, "->", f$value %% 100))
    fs <- freq(st)
    area_h <- setNames(fs$count * px, fs$value)
    cells <- cellFromRowCol(st, pts$row + 1, pts$col + 1)
    d <- data.frame(h = st[cells][, 1], m = pair[cells][, 1])
    d$m <- paste(d$m %/% 100, "->", d$m %% 100)
    d$r <- paste(pts$ref_from, "->", pts$ref_to)
    n_h <- table(d$h)
    d$w <- as.numeric(area_h[as.character(d$h)] / n_h[as.character(d$h)])
    classes <- sort(unique(c(names(X), d$r)))
    X <- X[classes]
    X[is.na(X)] <- 0
    names(X) <- classes
    des <- svydesign(ids = ~1, strata = ~h, weights = ~w, data = d)
    out <- list()
    for (k in classes) {
        # (update() looks the names up in the design's variables first: columns of d)
        d$y <- as.numeric(d$r == k)
        d$x <- as.numeric(d$m == k)
        d$yx <- d$y * d$x
        d$dd <- d$y - d$x
        des <- update(des, y = d$y, x = d$x, yx = d$yx, dd = d$dd)
        res <- list(mapped = X[[k]])
        if (estimator == "direct") {
            e <- svytotal(~y, des)
            res$area <- c(coef(e)[[1]], SE(e)[[1]])
        } else if (estimator == "difference") {
            e <- svytotal(~dd, des)
            res$area <- c(X[[k]] + coef(e)[[1]], SE(e)[[1]])
        } else if (sum(d$x) > 0) {
            e <- svyratio(~y, ~x, des)
            res$area <- c(coef(e)[[1]] * X[[k]], SE(e)[[1]] * X[[k]])
        } else {
            res$area <- c(NA, NA)
        }
        if (sum(d$x) > 0) {
            u <- svyratio(~yx, ~x, des)
            res$users <- c(coef(u)[[1]], SE(u)[[1]])
        } else {
            res$users <- c(NA, NA)
        }
        if (sum(d$y) > 0) {
            p <- svyratio(~yx, ~y, des)
            res$producers <- c(coef(p)[[1]], SE(p)[[1]])
        } else {
            res$producers <- c(NA, NA)
        }
        out[[k]] <- res
    }
    des <- update(des, agree = as.numeric(d$r == d$m))
    o <- svymean(~agree, des)
})[["elapsed"]]
write_json(list(survey = as.character(packageVersion("survey")), classes = out, overall = c(coef(o)[[1]], SE(o)[[1]])),
           file.path(args[1], paste0("survey_", estimator, ".json")), digits = NA, auto_unbox = TRUE, na = "null")
cat("SECONDS", s, "\n")
