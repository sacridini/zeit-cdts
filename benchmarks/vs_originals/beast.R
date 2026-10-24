# Rbeast 1.0.2 (Zhao et al. 2019; GPL >= 2) on the series of a binary file of float64 (pixels x dates, NaN missing):
# beast() for one series, beast123() (its batch, numParThreads threads) for several, with the arguments of beast()'s
# defaults (season harmonic, period 12, 2 or more chains from seed 42, the fast credible intervals). The call is
# timed; the trends and their numbers of changepoints are written next to the input for the check.
#
#   Rscript beast.R values.bin n_pixels n_dates samples chains threads cputype
suppressMessages(library(Rbeast))
a <- commandArgs(TRUE)
np <- as.integer(a[2]); nt <- as.integer(a[3]); samples <- as.integer(a[4]); chains <- as.integer(a[5])
threads <- as.integer(a[6]); cputype <- a[7]
Y <- matrix(readBin(a[1], "double", np * nt), nrow = np, byrow = TRUE)
mcmc <- list(seed = 42, samples = samples, chainNumber = chains, burnin = 200, thinningFactor = 5, maxMoveStepSize = 0,
             trendResamplingOrderProb = 0.1, seasonResamplingOrderProb = 0.17, credIntervalAlphaLevel = 0.95)
prior <- list(modelPriorType = 1, seasonMinOrder = 0, seasonMaxOrder = 5, seasonMinKnotNum = 0, seasonMaxKnotNum = 10,
              trendMinOrder = 0, trendMaxOrder = 1, trendMinKnotNum = 0, trendMaxKnotNum = 10, K_MAX = 0,
              precValue = 1.5, precPriorType = "componentwise")
extra <- list(dumpInputData = TRUE, computeCredible = TRUE, fastCIComputation = TRUE, computeSeasonOrder = TRUE,
              computeTrendOrder = TRUE, computeSeasonChngpt = TRUE, computeTrendChngpt = TRUE,
              computeSeasonAmp = TRUE, computeTrendSlope = TRUE, tallyPosNegSeasonJump = TRUE,
              tallyPosNegTrendJump = TRUE, tallyIncDecTrendJump = TRUE, quiet = TRUE, printProgress = FALSE,
              printParameter = FALSE, numParThreads = threads, numThreadsPerCPU = 1)
if (cputype != "default") extra$cputype <- cputype
meta <- list(whichDimIsTime = 2, period = 12, deltaTime = 1, startTime = 1, season = "harmonic")
invisible(capture.output(
  s <- system.time(o <- beast123(if (np == 1) Y[1, ] else Y, metadata = meta, prior = prior, mcmc = mcmc,
                                 extra = extra))[["elapsed"]]))
trend <- if (np == 1) matrix(o$trend$Y, nrow = 1) else o$trend$Y
out <- sub(".bin", paste0("_", cputype, "_out.bin"), a[1], fixed = TRUE)
writeBin(as.double(c(t(cbind(trend, o$trend$ncp)))), out)
cat("SECONDS", s, "\n")
