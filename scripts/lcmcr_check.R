## Fits LCMCR (Manrique-Vallier 2016) on the incidence matrices exported by lcmcr_check.py.
## One fit per (paradigm, truncation K, DP prior a_alpha=b_alpha); writes lcmcr_results.csv.
## Run from results/ (lcmcr_check.py sets the working directory), where the inputs are written.
suppressMessages(library(LCMCR))

fit_one <- function(X, K, a, seed = 1) {
  Xf <- as.data.frame(X)
  Xf[] <- lapply(Xf, function(col) factor(as.integer(col), levels = c(0, 1),
                                          labels = c("0", "1")))
  s <- lcmCR(captures = Xf, in_list_label = "1", not_in_list_label = "0",
             K = K, a_alpha = a, b_alpha = a, seed = seed,
             buffer_size = 20000, thinning = 10, verbose = FALSE)
  draws <- lcmCR_PostSampl(s, burnin = 10000, samples = 20000, thinning = 10,
                           output = FALSE)
  q <- quantile(draws, c(0.025, 0.5, 0.975))
  data.frame(N_lo = as.numeric(q[1]), N_med = as.numeric(q[2]),
             N_hi = as.numeric(q[3]), N_mean = mean(draws))
}

out <- list()
for (nm in c("sources", "authors", "centuries")) {
  X <- read.csv(sprintf("lcmcr_input_%s.csv", nm), check.names = FALSE)
  Sobs <- nrow(X)
  for (K in c(4, 6, 8)) {
    for (a in c(0.25, 1.0)) {
      r <- fit_one(X, K, a)
      out[[length(out) + 1]] <- cbind(
        data.frame(paradigm = nm, J = ncol(X), Sobs = Sobs, K = K, a_alpha = a), r)
      cat(sprintf("%-10s J=%3d Sobs=%3d K=%d a=%.2f -> N = %.0f [%.0f, %.0f]\n",
                  nm, ncol(X), Sobs, K, a, r$N_med, r$N_lo, r$N_hi))
    }
  }
}
res <- do.call(rbind, out)
write.csv(res, "lcmcr_results.csv", row.names = FALSE)
