# run-bootstrap-median.R
# Bootstrap Standard Error of the Median – R Reference Implementation
#
# This script mirrors the Go implementation in cmd/bootstrap/main.go and
# internal/bootstrap/bootstrap.go.  Run both programs on the same CSV data
# (exported by Go via -savedata flag) to produce directly comparable results.
#
# R packages used:
#   boot  – https://cran.r-project.org/package=boot   (Davison & Hinkley 1997)
#   bootstrap – https://cran.r-project.org/package=bootstrap (Efron & Tibshirani 1993)
#
# Install once:
#   install.packages(c("boot", "bootstrap"))
#
# Method: Bootstrapping and simulation-based inference
#   Gelman, A. & Vehtari, A. (2021). What are the most important statistical
#   ideas of the past 50 years? Journal of the American Statistical Association,
#   116(536), 2087–2097.  https://doi.org/10.1080/01621459.2021.1938081
#
# Usage:
#   Rscript run-bootstrap-median.R
# -------------------------------------------------------------------

library(boot)

set.seed(42)          # matches Go seed for comparison

N_OBS     <- 500      # observations per dataset
N_RESAMP  <- 1000     # bootstrap resamples (B)

# Estimator function: sample median (boot package statistic interface)
median_stat <- function(data, indices) {
  median(data[indices])
}

# ── Helper: run and report one bootstrap analysis ────────────────────────────
run_bootstrap <- function(label, data) {
  cat("\n  Distribution:", label, "\n")
  cat("  ─────────────────────────────────────────────────\n")

  mem_before <- gc(verbose = FALSE)

  t0 <- proc.time()
  boot_result <- boot(data        = data,
                      statistic   = median_stat,
                      R           = N_RESAMP,
                      sim         = "ordinary")
  elapsed <- proc.time() - t0

  mem_after <- gc(verbose = FALSE)

  ci <- boot.ci(boot_result, type = "perc", conf = 0.95)

  original_est  <- boot_result$t0
  boot_se       <- sd(boot_result$t)
  boot_bias     <- mean(boot_result$t) - original_est
  ci_lower      <- ci$percent[4]
  ci_upper      <- ci$percent[5]

  cat(sprintf("  Sample Median (original):  %8.4f\n", original_est))
  cat(sprintf("  Bootstrap SE of Median:    %8.4f\n", boot_se))
  cat(sprintf("  Bootstrap Bias:            %8.4f\n", boot_bias))
  cat(sprintf("  95%% Percentile CI:         [%7.4f, %7.4f]\n", ci_lower, ci_upper))
  cat(sprintf("  Bootstrap resamples:       %d\n",   N_RESAMP))
  cat(sprintf("  Elapsed (user+sys):        %.3f s\n",
              elapsed["user.self"] + elapsed["sys.self"]))

  # Return a named row for the summary table
  list(shape        = label,
       n            = N_OBS,
       B            = N_RESAMP,
       estimate     = original_est,
       se           = boot_se,
       bias         = boot_bias,
       ci_lower     = ci_lower,
       ci_upper     = ci_upper,
       elapsed_s    = as.numeric(elapsed["user.self"] + elapsed["sys.self"]))
}

# ── Generate datasets ────────────────────────────────────────────────────────
sym_data  <- rnorm(N_OBS, mean = 0, sd = 1)
pos_data  <- rlnorm(N_OBS, meanlog = 0, sdlog = 1)
neg_raw   <- rlnorm(N_OBS, meanlog = 0, sdlog = 1)
neg_data  <- max(neg_raw) - neg_raw   # reflect → left-skewed

# Save datasets so Go can load them for a true apples-to-apples comparison
dir.create("testdata", showWarnings = FALSE)
write.csv(data.frame(value = sym_data),  "testdata/symmetric.csv",         row.names = FALSE)
write.csv(data.frame(value = pos_data),  "testdata/positively_skewed.csv", row.names = FALSE)
write.csv(data.frame(value = neg_data),  "testdata/negatively_skewed.csv", row.names = FALSE)
cat("Test data written to testdata/\n")

# ── Run analyses ─────────────────────────────────────────────────────────────
cat("\n══════════════════════════════════════════════════════════════════════\n")
cat("  Bootstrap Standard Error of the Median – R (boot package)\n")
cat(sprintf("  n=%d observations  B=%d resamples  seed=42\n", N_OBS, N_RESAMP))
cat("══════════════════════════════════════════════════════════════════════\n")

results <- list(
  run_bootstrap("Symmetric (Normal)",                    sym_data),
  run_bootstrap("Positively Skewed (Log-Normal)",        pos_data),
  run_bootstrap("Negatively Skewed (Reflected Log-Normal)", neg_data)
)

# ── Theoretical SE for symmetric case ───────────────────────────────────────
theoretical_se <- sqrt(pi / 2) / sqrt(N_OBS)
cat(sprintf("\n  Theoretical SE of median (Normal, n=%d): %.4f\n",
            N_OBS, theoretical_se))

# ── Save summary CSV ─────────────────────────────────────────────────────────
dir.create("results", showWarnings = FALSE)
summary_df <- do.call(rbind, lapply(results, as.data.frame))
write.csv(summary_df, "results/r_results.csv", row.names = FALSE)
cat("\n  R results saved to results/r_results.csv\n")

cat("\n══════════════════════════════════════════════════════════════════════\n\n")
