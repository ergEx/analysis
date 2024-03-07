#!/usr/bin/env Rscript
library("optparse")
library("BAS")
library(stargazer)
# Supressing warnings
options(warn=-1)

## 
find_indices <- function(lst, value) {
  sapply(lst, function(x) any(x == value))
}



bas_analysis <- function(target, dataset, title_var, output_file){
  bas_model <- bas.lm(paste0(target, ' ~ .'), data=dataset, method="BAS")
  bas_coef <- coef(bas_model)
  
  factors <- bas_model$namesx
  inclusion_prob <- bas_model$probne0
  posterior_mean <- bas_coef$postmean
  posterior_sd <- bas_coef$postsd
  
  indices_list <- lapply(c(0 : (length(bas_model$namesx) - 1)), function(value) find_indices(bas_model$which, value))
  
  prior_val <- c()
  bfs <- c()
  
  for (ii in c(1 : length(bas_model$namesx)))
  {
    prior_val[ii] <- (sum(bas_model$priorprobs[unlist(indices_list[ii])]) / (1 - sum(bas_model$priorprobs[unlist(indices_list[ii])])))
    bfs[ii] <-(sum(bas_model$postprobs[unlist(indices_list[ii])]) / (1 - sum(bas_model$postprobs[unlist(indices_list[ii])]))) / prior_val[ii]
  }
  
  sum_df <- data.frame(independvar = factors, inclusion_rob = inclusion_prob, posterior_mean = round(posterior_mean, 4),
                       posterior_sd = round(posterior_sd, 4), prior=round(prior_val, 4), bf = round(bfs, 4))
  
  new_col_names <- c("Independent Variable", "P(B != 0 | Y)", "posterior mean", "posterior SD", "Prior", "BF")  
  bf[1] <- ' '
  colnames(sum_df) <- new_col_names
  
  sum_df_lat  <- stargazer(sum_df, summary=FALSE, title=title_var)
  
  fileConn<-file(output_file)
  writeLines(sum_df_lat, fileConn)
  close(fileConn)
}

set.seed(2023)

path = ''

confounds <- read.delim2(paste(path, 'full_data_anova.tsv', sep=''), sep='\t', header=TRUE, dec='.')
explanation <- read.delim2(paste(path, 'full_data_regression.tsv', sep=''), sep='\t', header=TRUE, dec='.')
explanation$Sex <- as.factor(explanation$Sex)

bas_analysis('d_eta', confounds, 'regression of confounding variables', '../paper_figures/regression1.txt')
bas_analysis('X0.0_partial_pooling', explanation, 'regression on additive risk-preferences', '../paper_figures/regression2.txt')
bas_analysis('X1.0_partial_pooling', explanation, 'regression on multiplicative risk-preferences', '../paper_figures/regression3.txt')