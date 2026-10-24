# sfdep's emerging_hotspot_analysis on a grid of cells (queen neighbours) read from a CSV of values (time, row,
# column in the order of st_make_grid's cells): the call timed; the Gi* written next to the input for the check.
args <- commandArgs(TRUE)
input <- args[1]; nx <- as.integer(args[2]); ny <- as.integer(args[3]); nt <- as.integer(args[4])
nsim <- if (length(args) > 4) as.integer(args[5]) else 199L
suppressMessages({library(sfdep); library(sf)})
v <- scan(input, quiet = TRUE)
g <- st_make_grid(st_bbox(c(xmin = 0, ymin = 0, xmax = nx, ymax = ny)), n = c(nx, ny))
geo <- st_sf(id = sprintf("%06d", seq_along(g)), geometry = g)
df <- expand.grid(id = geo$id, t = seq_len(nt), stringsAsFactors = FALSE)
df$value <- v
df$time <- as.Date("2000-01-01") + df$t * 365
st <- spacetime(df, geo, .loc_col = "id", .time_col = "time")
set.seed(1)
s <- system.time(e <- emerging_hotspot_analysis(st, "value", k = 1, nsim = nsim, include_gi = TRUE))[["elapsed"]]
out <- as.data.frame(st)
gi <- attr(e, "gi_star")$gi_star[order(out$t, out$id)]
writeLines(format(gi, digits = 17), sub(".txt", "_gi.txt", input, fixed = TRUE))
cat("SECONDS", s, "\n")
