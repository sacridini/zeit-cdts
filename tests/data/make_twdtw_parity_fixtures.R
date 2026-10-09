## Reference distances of R's twdtw (1.0.1) for tests/test_twdtw_api.py:
## python make_twdtw_parity_fixtures.py, then Rscript make_twdtw_parity_fixtures.R in this
## folder, then merge r_distances.json into twdtw_r_parity.json (key "r_distance" per case).
suppressMessages(library(twdtw))
suppressMessages(library(jsonlite))
cases <- fromJSON("cases.json", simplifyVector = FALSE)
frame <- function(s) {
  v <- s$values
  if (is.list(v[[1]])) {
    m <- do.call(rbind, lapply(v, function(r) sapply(r, function(e) if (is.null(e)) NA_real_ else e)))
  } else {
    m <- matrix(sapply(v, function(e) if (is.null(e)) NA_real_ else e), ncol = 1)
  }
  df <- data.frame(time = as.Date(unlist(s$time)))
  for (b in seq_len(ncol(m))) df[[paste0("b", b)]] <- m[, b]
  df
}
out <- list()
for (cs in cases) {
  x <- frame(cs$series)
  d <- list()
  for (name in names(cs$patterns)) {
    y <- frame(cs$patterns[[name]])
    d[[name]] <- twdtw(x = x, y = y, time_weight = c(cs$steep, cs$mid), cycle_length = "year",
                       time_scale = "day", index_column = "time", output = "distance")
    d[[name]] <- as.numeric(d[[name]])
  }
  out[[cs$name]] <- d
}
cat(toJSON(out, digits = NA, auto_unbox = TRUE), file = "r_distances.json")
cat("ok\n")
