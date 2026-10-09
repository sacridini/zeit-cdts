# Fixtures for tests/test_accuracy.py: sits 1.5.4's area-weighted accuracy
# (sits:::.accuracy_area_assess, Olofsson et al. 2014) on given error matrices and class
# areas. The function's own body runs; only the cube it would read the areas from is
# replaced by the areas themselves.
#
#   Rscript tests/data/make_accuracy_sits.R
library(sits)
library(jsonlite)

assess <- function(pred, ref, labels, area) {
    f <- sits:::.accuracy_area_assess
    env <- new.env(parent = environment(f))
    env$.check_set_caller <- function(...) invisible(NULL)
    env$.check_is_class_cube <- function(...) invisible(NULL)
    env$.cube_labels <- function(...) labels
    env$.cube_class_areas <- function(...) area
    environment(f) <- env
    f(NULL, pred, ref)
}

cases <- list()
add_case <- function(name, labels, matrix, area) {
    pred <- rep(rep(labels, ncol(matrix)), as.vector(matrix))
    ref <- rep(rep(labels, each = nrow(matrix)), as.vector(matrix))
    names(area) <- labels
    res <- assess(pred, ref, labels, area)
    cases[[name]] <<- list(
        labels = labels, matrix = matrix, area = unname(area),
        adjusted_area = unname(res$error_ajusted_area[labels]),
        stderr_area = unname(res$stderr_area[labels]),
        user = unname(res$accuracy$user[labels]),
        producer = unname(res$accuracy$producer[labels]),
        overall = res$accuracy$overall
    )
}

# Olofsson et al. (2014), section 5: map rows, reference columns, areas in pixels
add_case("olofsson2014", c("a_deforestation", "b_forest_gain", "c_stable_forest", "d_stable_nonforest"),
         matrix(c(66, 0, 5, 4, 0, 55, 8, 12, 1, 0, 153, 11, 2, 1, 9, 313), nrow = 4, byrow = TRUE),
         c(200000, 150000, 3200000, 6450000))

# A three-class map with a rare class and errors everywhere
add_case("three_classes", c("a_change", "b_forest", "c_other"),
         matrix(c(41, 6, 3, 4, 88, 8, 2, 7, 141), nrow = 3, byrow = TRUE),
         c(5200, 61000, 133800))

write_json(cases, "tests/data/accuracy_sits.json", digits = NA, auto_unbox = TRUE, pretty = TRUE)
