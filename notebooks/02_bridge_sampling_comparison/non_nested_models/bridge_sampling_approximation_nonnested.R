library(tidyverse)
library(rstan)
library(bridgesampling)
library(bayesplot)
library(posterior)
library(reticulate)
np <- import("numpy")


### Set working directory to main folder
setwd(dirname(dirname(dirname(dirname(rstudioapi::getSourceEditorContext()$path)))))


### Utility functions
save_parameter_estimates <- function(row, stanfits, parameter_estimates, test_data_true_indices){
  # loops through model fits to save parameter estimates
  for (j in 1:length(stanfits)){
    # prepare rstan output
    fit_means <- colMeans(stanfits[[j]])
    
    # write general information in matrix
    parameter_estimates[[j]][row,"dataset"] <- i
    parameter_estimates[[j]][row,"true_model"] <- test_data_true_indices[i]
    
    # write hit & false alarm probability estimates in matrix
    hit_estimates <- fit_means[match(sprintf("p_h_m[%s]", 1), names(fit_means))
                                 :match(sprintf("p_h_m[%s]", dim(test_data_single)[1]), names(fit_means))]
    parameter_estimates[[j]][row,"mean(p_h_m)"] <- mean(hit_estimates)
    parameter_estimates[[j]][row,"var(p_h_m)"] <- var(hit_estimates)

    fa_estimates <- fit_means[match(sprintf("p_f_m[%s]", 1), names(fit_means))
                                 :match(sprintf("p_f_m[%s]", dim(test_data_single)[1]), names(fit_means))]
    parameter_estimates[[j]][row,"mean(p_f_m)"] <- mean(fa_estimates)
    parameter_estimates[[j]][row,"var(p_f_m)"] <- var(fa_estimates)
    
    # write other parameter estimates in matrix
    if ("mu_h" %in% colnames(parameter_estimates[[j]])){ # SDT model
      parameter_estimates[[j]][row,"mu_h"] <- fit_means["mu_h"]
      parameter_estimates[[j]][row,"sigma_h"] <- fit_means["sigma_h"]
      parameter_estimates[[j]][row,"mu_f"] <- fit_means["mu_f"]
      parameter_estimates[[j]][row,"sigma_f"] <- fit_means["sigma_f"]      
    }
    if ("mu_d" %in% colnames(parameter_estimates[[j]])){ # MPT model
      parameter_estimates[[j]][row,"mu_d"] <- fit_means["mu_d"]
      parameter_estimates[[j]][row,"mu_g"] <- fit_means["mu_g"]
      parameter_estimates[[j]][row,"lambdas_1"] <- fit_means["lambdas[1]"]
      parameter_estimates[[j]][row,"lambdas_2"] <- fit_means["lambdas[2]"]
      parameter_estimates[[j]][row,"Q[1,1]"] <- fit_means["Q[1,1]"]
      parameter_estimates[[j]][row,"Q[1,2]"] <- fit_means["Q[1,2]"] 
      parameter_estimates[[j]][row,"Q[2,1]"] <- fit_means["Q[2,1]"] 
      parameter_estimates[[j]][row,"Q[2,2]"] <- fit_means["Q[2,2]"]           
    }
  }
  return(parameter_estimates)
}

### Import test data sets
test_data <- np$load("data/02_bridge_sampling_comparison/non_nested_models/test_data.npy", allow_pickle = TRUE)[[1]][[1]]
test_data_true_indices <- np$load("data/02_bridge_sampling_comparison/non_nested_models/test_data_true_indices.npy")


##### Bayes factor computation ######

### prepare result matrices

## prepare matrices for parameter recovery results
column_names_pars <- c("dataset","true_model","mean(p_h_m)","var(p_h_m)","mean(p_f_m)","var(p_f_m)",
                       "mu_h","sigma_h","mu_f","sigma_f","mu_d","mu_g","lambdas_1","lambdas_2",
                       "Q[1,1]","Q[1,2]","Q[2,1]","Q[2,2]")
# SDT model
parameter_estimates_sdt <- matrix(nrow = dim(test_data)[1], ncol = length(column_names_pars))
colnames(parameter_estimates_sdt) <- column_names_pars
# MPT model
parameter_estimates_mpt <- matrix(nrow = dim(test_data)[1], ncol = length(column_names_pars))
colnames(parameter_estimates_mpt) <- column_names_pars
# store
parameter_estimates <- list(parameter_estimates_sdt, parameter_estimates_mpt)

## prepare matrix for comparison results
column_names_comp <- c("dataset","true_model","m0_prob","m1_prob","selected_model",
                       "bayes_factor", "m0_bridge_error", "m1_bridge_error", 
                       "compile_time", "stan_time", "bridge_time") # m0 = sdt / m1 = mpt
comparison_results <- matrix(nrow = dim(test_data)[1], ncol = length(column_names_comp))
colnames(comparison_results) <- column_names_comp

## prepare matrix to log number of divergent transitions
n_div_transitions <- matrix(nrow = 100, ncol = 2)


### Specify Stan models

# model file paths
sdt_model_file <-"notebooks/02_bridge_sampling_comparison/non_nested_models/sdt_model.stan"
mpt_model_file <-"notebooks/02_bridge_sampling_comparison/non_nested_models/mpt_model.stan"

# measure time
compile_start <- Sys.time()

# Compile models
sdt_model <- stan_model(file = sdt_model_file)
mpt_model <- stan_model(file = mpt_model_file)

# Measure time
compile_end <- Sys.time()
compile_time <- difftime(compile_end, compile_start, unit="secs")
print(compile_time)

### Loop over test data sets

for (i in 1:dim(test_data)[1]){    
  
  # select data set
  test_data_single <- test_data[i,,,]
  
  # Measure stan start time
  stan_start <- Sys.time()
  
  
  ### Fit models
  
  # Prepare data for sampling
  sdt_data <- list(
    M = dim(test_data_single)[1],  # Number of clusters.
    N = dim(test_data_single)[2],  # Number of observations.
    N_old_new = dim(test_data_single)[2]/2, # Number of old/new items, equal as 50/50 proportion of old and new items is given
    X = test_data_single[,,2] # 2D array of observations (without condition indicator).
  )
  
  mpt_data <- list(
    M = dim(test_data_single)[1],  # Number of clusters.
    N = dim(test_data_single)[2],  # Number of observations.
    N_old_new = dim(test_data_single)[2]/2, # Number of old/new items, equal as 50/50 proportion of old and new items is given
    X = test_data_single[,,2] # 2D array of observations (without condition indicator).
  )

  
  # Fit
  sdt_fit <- sampling(sdt_model, data = sdt_data, iter = 50000, warmup = 1000, chains = 4, cores = 4) #control=list(adapt_delta=0.99))
  mpt_fit <- sampling(mpt_model, data = mpt_data, iter = 50000, warmup = 1000, chains = 4, cores = 4) #control=list(adapt_delta=0.99))
  # adapt_delta=0.99 leads to less divergent transitions, but doubles sampling time while not improving approximation performance
  
  # Measure Stan end time
  stan_end <- Sys.time()
  
  # Save parameter estimates
  stanfits <- list(as.matrix(sdt_fit),as.matrix(mpt_fit))
  
  parameter_estimates <- save_parameter_estimates(i, stanfits, parameter_estimates, test_data_true_indices)
  
  # Optional: Save div. transitions to explore their impact
  n_div_transitions[i,1] <- sum(subset(nuts_params(sdt_fit), Parameter == "divergent__")$Value)
  n_div_transitions[i,2] <- sum(subset(nuts_params(mpt_fit), Parameter == "divergent__")$Value)
  
  ### Bridge Sampling
  
  # Measure Bridge Sampling start time
  bridge_start <- Sys.time()
  
  # Compute the (Log) Marginal Likelihoods
  sdt_marginal <- bridge_sampler(sdt_fit, silent = TRUE)
  mpt_marginal <- bridge_sampler(mpt_fit, silent = TRUE)
  
  # Compute Bayes factors in favor of mpt model
  bayes_factor_for_mpt <- bf(mpt_marginal, sdt_marginal) 
  
  # Compute posterior model probabilities (assuming equal prior model probabilities)
  post_probs <- post_prob(sdt_marginal, mpt_marginal)
  
  # Compute selected model
  selected_model <- ifelse(post_probs[1] > 0.5, 0, 1)
  
  # Compute approximate percentage error of marginal likelihood estimates
  sdt_bridge_error <- error_measures(sdt_marginal)$percentage
  mpt_bridge_error <- error_measures(mpt_marginal)$percentage
  
  # Measure Bridge Sampling end time
  bridge_end <- Sys.time()
  
  # Calculate computation times
  stan_time <- difftime(stan_end, stan_start, unit="secs")
  bridge_time <- difftime(bridge_end, bridge_start, unit="secs")
  
  # Save comparison results
  comparison_results[i,"dataset"] <- i
  comparison_results[i,"true_model"] <- test_data_true_indices[i,2]
  comparison_results[i,"m0_prob"] <- post_probs[1] # m0 = sdt
  comparison_results[i,"m1_prob"] <- post_probs[2] # m1 = mpt  
  comparison_results[i,"selected_model"] <- selected_model
  comparison_results[i,"bayes_factor"] <- bayes_factor_for_mpt$bf
  comparison_results[i,"m0_bridge_error"] <- sdt_bridge_error
  comparison_results[i,"m1_bridge_error"] <- mpt_bridge_error
  comparison_results[i,"compile_time"] <- compile_time
  comparison_results[i,"stan_time"] <- stan_time 
  comparison_results[i,"bridge_time"] <- bridge_time
  
  # Print progress
  print(sprintf("Dataset %s successfully finished.", i))
  print(sprintf("Stan time: %s", stan_time))
  print(sprintf("Bridge Sampling time: %s", bridge_time))
}

### Check output
parameter_estimates
comparison_results

### Export experimental results
exp_time <- format(Sys.time(), '%Y_%m_%d')
exp_file_params <- sprintf("data/02_bridge_sampling_comparison/non_nested_models/%s_BF_BS_params", exp_time)
exp_file_comp <- sprintf("data/02_bridge_sampling_comparison/non_nested_models/%s_BF_BS", exp_time)

write.table(parameter_estimates, file=exp_file_params)
write.table(comparison_results, file=exp_file_comp)

# Optional: Export number of divergent transitions per fit
n_div_t_file = sprintf("data/02_bridge_sampling_comparison/non_nested_models/%s_BF_BS_n_div_trans", exp_time)
write.table(n_div_transitions, file=n_div_t_file)


### Load results of earlier experiments
old_params <- read.table("data/02_bridge_sampling_comparison/non_nested_models/2022_05_03_BF_BS_params")
old_comp <- read.table("data/02_bridge_sampling_comparison/non_nested_models/2022_05_03_BF_BS")



##### Diagnostics #####
# Inspect single stanfit objects

### Inspect traceplots
traceplot(sdt_fit, pars = c("mu_h", "sigma_h", "mu_f", "sigma_f"))
traceplot(mpt_fit, pars = c("mu_d", "mu_g", "lambdas[1]", "lambdas[2]",
                            "Q[1,1]", "Q[1,2]", "Q[2,1]", "Q[2,2]"))

### Convential n_eff and Rhat statistics
print(sdt_fit) 
print(mpt_fit)

#### Improved Rhat and ess statistics

## SDT model
# mu_h
Rhat(extract_variable_matrix(sdt_fit, "mu_h"))
ess_bulk(extract_variable_matrix(sdt_fit, "mu_h"))
ess_tail(extract_variable_matrix(sdt_fit, "mu_h"))

# sigma_h
Rhat(extract_variable_matrix(sdt_fit, "sigma_h"))
ess_bulk(extract_variable_matrix(sdt_fit, "sigma_h"))
ess_tail(extract_variable_matrix(sdt_fit, "sigma_h"))

# mu_f
Rhat(extract_variable_matrix(sdt_fit, "mu_f"))
ess_bulk(extract_variable_matrix(sdt_fit, "mu_f"))
ess_tail(extract_variable_matrix(sdt_fit, "mu_f"))

# sigma_f
Rhat(extract_variable_matrix(sdt_fit, "sigma_f"))
ess_bulk(extract_variable_matrix(sdt_fit, "sigma_f"))
ess_tail(extract_variable_matrix(sdt_fit, "sigma_f"))

## MPT model
# mu_d
Rhat(extract_variable_matrix(mpt_fit, "mu_d"))
ess_bulk(extract_variable_matrix(mpt_fit, "mu_d"))
ess_tail(extract_variable_matrix(mpt_fit, "mu_d"))

# mu_g
Rhat(extract_variable_matrix(mpt_fit, "mu_g"))
ess_bulk(extract_variable_matrix(mpt_fit, "mu_g"))
ess_tail(extract_variable_matrix(mpt_fit, "mu_g"))

# lambdas[1]
Rhat(extract_variable_matrix(mpt_fit, "lambdas[1]"))
ess_bulk(extract_variable_matrix(mpt_fit, "lambdas[1]"))
ess_tail(extract_variable_matrix(mpt_fit, "lambdas[1]"))

# lambdas[2]
Rhat(extract_variable_matrix(mpt_fit, "lambdas[2]"))
ess_bulk(extract_variable_matrix(mpt_fit, "lambdas[2]"))
ess_tail(extract_variable_matrix(mpt_fit, "lambdas[2]"))

# Q[1,1]
Rhat(extract_variable_matrix(mpt_fit, "Q[1,1]"))
ess_bulk(extract_variable_matrix(mpt_fit, "Q[1,1]"))
ess_tail(extract_variable_matrix(mpt_fit, "Q[1,1]"))

# Q[1,2]
Rhat(extract_variable_matrix(mpt_fit, "Q[1,2]"))
ess_bulk(extract_variable_matrix(mpt_fit, "Q[1,2]"))
ess_tail(extract_variable_matrix(mpt_fit, "Q[1,2]"))

# Q[2,1]
Rhat(extract_variable_matrix(mpt_fit, "Q[2,1]"))
ess_bulk(extract_variable_matrix(mpt_fit, "Q[2,1]"))
ess_tail(extract_variable_matrix(mpt_fit, "Q[2,1]"))

# Q[2,2]
Rhat(extract_variable_matrix(mpt_fit, "Q[2,2]"))
ess_bulk(extract_variable_matrix(mpt_fit, "Q[2,2]"))
ess_tail(extract_variable_matrix(mpt_fit, "Q[2,2]"))

