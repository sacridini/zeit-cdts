# CAST's trainDI and aoa (CAST 1.0.4) on training points and new data read from CSVs (no header): the calls timed
# (trainDI, then aoa with it and LPD), the DI written next to the new data for the check.
args <- commandArgs(TRUE)
train <- read.csv(args[1], header = FALSE); newdata <- read.csv(args[2], header = FALSE)
names(newdata) <- names(train)
suppressMessages(library(CAST))
s1 <- system.time(tdi <- suppressMessages(trainDI(train = train, verbose = FALSE)))[["elapsed"]]
s2 <- system.time(a <- suppressMessages(aoa(newdata, train = train, trainDI = tdi, LPD = as.logical(args[3]),
                                              verbose = FALSE)))[["elapsed"]]
writeLines(format(a$DI, digits = 17), sub(".csv", "_di.txt", args[2], fixed = TRUE))
cat("TRAIN", s1, "\n")
cat("SECONDS", s1 + s2, "\n")
