# sits' Bayesian smoothing of a block (sits 1.5.4's .smooth_fn_bayes: the logits, bayes_smoother_fraction, back to
# probabilities) of probabilities read from a CSV (pixels x classes, row-major pixels): the call timed, the result
# written next to the input for the check.
args <- commandArgs(TRUE)
p <- as.matrix(read.csv(args[1], header = FALSE)); nrows <- as.integer(args[2]); ncols <- as.integer(args[3])
window <- as.integer(args[4]); smoothness <- as.numeric(args[5])
suppressMessages(library(sits))
fn <- get(".smooth_fn_bayes", envir = asNamespace("sits"))(window_size = window, neigh_fraction = 0.5,
                                                          smoothness = rep(smoothness, ncol(p)))
s <- system.time(out <- fn(values = p, block = list(nrows = nrows, ncols = ncols)))[["elapsed"]]
write.table(out, sub(".csv", "_out.csv", args[1], fixed = TRUE), sep = ",", row.names = FALSE, col.names = FALSE)
cat("SECONDS", s, "\n")
