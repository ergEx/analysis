#!/usr/bin/env Rscript
library("optparse")
library("BayesFactor")
library("lsr")

# Supressing warnings
options(warn=-1)

sequential_bayes_x_larger_y <- function(x_data, y_data, hypo){
  #' Performs a sequential BayesFactor analysis for the ergEx project.
  #' We perform a paired Bayesian one sided t-test on the effect size of
  #' x > y.
  #' Normally, we return the BFs for the test: H1: 0 < d < Inf, H0: d = 0
  #' If hypo = Q1, we test if H1: 0 < d < Inf, H0: !(0 < d < Inf).
  #'
  #' @x_data input vector one, testing if x > y.
  #' @y_data input vector two, testing if x > y
  #' @hypo characterstring, to see which hypothesis is tested (see above).
  #'
  #' Returns a DF in longform, with columns bf10, bf01, nsubs, scale.
  #' Where "bf10" and "bf01" are the bayesfactor in favor of H1 or H0, "nsubs" is the
  #' number of subjects the test was performed on (sequentially increasing from
  #' from 2 to length(x_data)), and finally "scale", the JSZ prior scale that has
  #' been used for the tests.
  #'

  nsubs <- length(x_data)
  out <- as.data.frame(matrix(data=NA, nrow = nsubs-1, ncol = 5))
  colnames(out)<- c("bf10","bf01","nsubs", "scale", 'cohens_d')

  cc <- 1

  for (sc in c('medium', 'wide', 'ultrawide')){

      for (x in 2 : nsubs)
      {
        # Our test here is: is the effect size in the range (0 < d < Inf). Against the null hypothesis d=0
        # See:
        bf <- ttestBF(x=x_data[1 : x], y=y_data[1 : x], paired=TRUE, nullInterval = c(0, Inf), rscale=sc)
        tmpbf <- extractBF(bf, onlybf = TRUE)

        # Packaging
        out[cc, 1 : 2] <- tmpbf
        out[cc, 3] <- x
        out[cc, 4] <- sc
        out[cc, 5] <- cohensD(x=x_data[1 : x], y=y_data[1 : x], method='paired')
        cc <- cc + 1
      }
  }
  return(out)
}

reporting_ttest_x_larger_y <- function(x, y, estim, hypo, iterations=100000){
  #' Performs a BayesFactor t-test for the ergEx project.
  #' We perform a paired Bayesian one sided t-test on the effect size of
  #' x > y.
  #' Normally, we return the BFs for the test: H1: 0 < d < Inf, H0: d = 0
  #' If hypo = Q1, we test if H1: 0 < d < Inf, H0: !(0 < d < Inf).
  #'
  #' @x input vector one, testing if x > y.
  #' @y input vector two, testing if x > y
  #' @estim how x and y were estimated.
  #' @hypo characterstring, to see which hypothesis is tested (see above).
  #' @iterations Number of iterations to sample from the posterior
  #'
  #' Returns a one row Df, with fields:
  #' "hypo", to code for the hypothesis
  #' "estim", the estimation procedure
  #' "BF10", the BayesFactor for the alternative,
  #' "mean_diff", the mean of x - y
  #' "sd_diff", the sd of x - y
  #' "mu_median", the median of the posterior on the difference (x-y)
  #' "mu_bci_025", the 2.5 % percentile of the posterior
  #' "mu_bci_975", the 97.5 % percentile of the posterior
  #' "delta_median", the median of the posterior on the effectsize of x - y.
  #' "delta_bci_025", the 2.5 % percentile of the posterior
  #' "delta_bci_975", the 97.5 % percentile of the posterior
  #'

  out <- as.data.frame(matrix(data=NA, nrow = 1, ncol = 12))
  colnames(out)<- c("hypo", "estimation", "BF10", "mean_diff","sd_diff", "mu_median", "mu_bci_025",
                    "mu_bci_975", "delta_median", "delta_bci_025", "delta_bci_975", 'cohens_d')

  diff <- x - y # Calculating mean difference
  test <- ttestBF(x=x, y=y, paired=TRUE, nullInterval = c(0, Inf), rscale='medium')

  # Getting samples from posterior for comparison Inf > d > 0, i.e. index 1.
  post <- posterior(test, index=1, iterations=iterations)

  # Packing into DF:
  out[1,1] <- hypo
  out[1,2] <- estim
  out[1,3] <- extractBF(test, onlybf=TRUE)[1] # Getting BayesFactor from index 1
  out[1,4] <- mean(diff) # classic descriptives of paired difference
  out[1,5] <- sd(diff)
  out[1,6:8] <- quantile(post[, 1], probs=c(0.5, 0.025,  0.975)) # Getting quantiles of posterior samples for mu (mean difference)
  out[1,9:11] <- quantile(post[, 3], probs=c(0.5, 0.025,  0.975)) # Getting quantiles of posterior sampels for delta (effect size)
  out[1, 12] <- cohensD(x=x, y=y, method='paired')

  return(out)
}


reporting_correlation_r_greater_0 <- function(x, y, estim, hypo, iterations=100000){
  #' Performs a BayesFactor correlation test for the ergEx project.
  #' We perform a one sided correlation of x and y.
  #'
  #' @x input vector one, testing if x > y.
  #' @y input vector two, testing if x > y
  #' @estim how x and y were estimated.
  #' @hypo characterstring, to see which hypothesis is tested (see above).
  #' @iterations Number of iterations to sample from the posterior
  #'
  #' Returns a one row Df, with fields:
  #' "hypo", to code for the hypothesis,
  #' "estim", the estimation procedure.
  #' "BF10", the BayesFactor for the alternative,
  #' "r_median", the median of the posterior on the correlation coefficition of x, y.
  #' "r_bci_025", the 2.5 % percentile of the posterior
  #' "r_bci_975", the 97.5 % percentile of the posterior
  #'
  out <- as.data.frame(matrix(data=NA, nrow=1, ncol=7))
  colnames(out) <- c('hypo', 'estimation', 'BF10', 'r_median', 'r_bci_025', 'r_bci_975', 'pearson_r')
  test <- correlationBF(x, y, nullInterval = c(0, 1), rscale='medium')
  post <- posterior(test, 1, iterations = iterations)
  out[1, 1] <- hypo
  out[1, 2] <- estim
  out[1, 3] <- extractBF(test, onlybf=TRUE)[1]
  out[1, 4: 6] <- quantile(post[, 1], probs=c(0.5, 0.025, 0.975))
  out[1, 7] <- cor.test(x, y)$estimate[1]

  return(out)
}


option_list = list(
  make_option(c('-p', '--path'), type='character', default='data/1_pilot/', help='Folder for jasp_input and for write out.', metavar='character'),
  make_option(c('-m', '--mode'), type='character', default='partial_pooling', help='The estimation method to run the tests on.', metavar = 'character')
)

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

path <- opt$path
mode <- opt$mode

print(getwd())

jasp_data <- read.delim2(paste(path, 'jasp_input.csv', sep=''), sep='\t', header=TRUE, dec='.')

# ================================ Hypothesis 1 ================================
# Our test for hypothesis 1 is if d_h0 > d_h1. More specifically in BayesFactor package terms:
# it is 0 < d(d_h0 - d_h1) < Inf vs !(0 < d(d_h0 - d_h1) < Inf)

d_h0 <- jasp_data[, paste('d_h0_', mode, sep='')]
d_h1 <- jasp_data[, paste('d_h1_', mode, sep='')]

q1_ttest <- reporting_ttest_x_larger_y(d_h0, d_h1, mode, 'Q1')
q1_sequential <- sequential_bayes_x_larger_y(d_h0, d_h1, 'Q1')

write.table(q1_sequential, paste(path, 'q1_sequential_', mode, '.csv', sep=''), sep='\t', row.names = FALSE)
# ================================ Hypothesis 2 ================================
# Our test for hypothesis 2 is if x_10 > x_00. More specifically in BayesFactor package terms:
# it is 0 < d(x_10 - x_00) < Inf vs d = 0.

x_00 <- jasp_data[, paste('X0.0_', mode, sep='')]
x_10 <- jasp_data[, paste('X1.0_', mode, sep='')]

q2_ttest <- reporting_ttest_x_larger_y(x_10, x_00, mode, 'Q2')
q2_sequential <- sequential_bayes_x_larger_y(x_10, x_00, 'Q2')

write.table(q2_sequential, paste(path, 'q2_sequential_', mode, '.csv', sep=''), sep='\t', row.names = FALSE)
# ================================ Hypothesis 3 ================================
# Our test for hypothesis 3 is if r(x_10, x_00) is larger > 0. More specifically in BayesFactor package terms:
# it is 0 < r(x_10, x_00) < Inf vs r = 0.

q3_corr <- reporting_correlation_r_greater_0(x_00, x_10, mode, 'Q3')

result_file <- file(paste(path, 'reporting_results_', mode, '.txt', sep=''), 'w')
write.table(q1_ttest, result_file, quote = TRUE, row.names = FALSE, sep='\t')
write.table(q2_ttest, result_file, quote = TRUE, row.names = FALSE, sep='\t')
write.table(q3_corr, result_file, quote = TRUE, row.names = FALSE, sep='\t')

close(result_file)

print("Statistical analysis completed.")