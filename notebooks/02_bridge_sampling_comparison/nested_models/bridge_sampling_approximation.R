library(tidyverse)
library(rstan)
library(bridgesampling)
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
    
    # write parameter estimates in matrix
    parameter_estimates[[j]][row,"dataset"] <- i
    parameter_estimates[[j]][row,"true_model"] <- test_data_true_indices[i]
    
    # write theta estimates in matrix
    # ugly since theta positions vary between model fit objects
    theta_estimates <- fit_means[match(sprintf("theta[%s]", 1), names(fit_means))
                                 :match(sprintf("theta[%s]", dim(test_data_single)[1]), names(fit_means))]
    parameter_estimates[[j]][row,"mean(theta)"] <- mean(theta_estimates)
    parameter_estimates[[j]][row,"var(theta)"] <- var(theta_estimates)
    
    # write other parameter estimates in matrix
    parameter_estimates[[j]][row,"sigma2"] <- fit_means["sigma2"]
    if ("mu" %in% colnames(parameter_estimates[[j]])){
      parameter_estimates[[j]][row,"mu"] <- fit_means["mu"]      
    }
    parameter_estimates[[j]][row,"tau2"] <- fit_means["tau2"]
    
  }
  return(parameter_estimates)
}


### Import test data sets
test_data <- np$load("data/02_bridge_sampling_comparison/nested_models/test_data.npy", allow_pickle = TRUE)[[1]][[1]]
test_data_true_indices <- np$load("data/02_bridge_sampling_comparison/nested_models/test_data_true_indices.npy")



##### Bayes factor computation ######


### prepare result matrices

# prepare matrices for parameter recovery results
column_names_pars <- c("dataset","true_model","mean(theta)","var(theta)",
                       "sigma2","mu","tau2")
# m0
parameter_estimates_m0 <- matrix(nrow = dim(test_data)[1], ncol = length(column_names_pars))
colnames(parameter_estimates_m0) <- column_names_pars
# m1
parameter_estimates_m1 <- matrix(nrow = dim(test_data)[1], ncol = length(column_names_pars))
colnames(parameter_estimates_m1) <- column_names_pars
# store
parameter_estimates <- list(parameter_estimates_m0, parameter_estimates_m1)

# prepare matrix for comparison results
column_names_comp <- c("dataset","true_model","m0_prob","m1_prob","selected_model",
                       "bayes_factor", "m0_bridge_error", "m1_bridge_error", 
                       "compile_time", "stan_time", "bridge_time")
comparison_results <- matrix(nrow = dim(test_data)[1], ncol = length(column_names_comp))
colnames(comparison_results) <- column_names_comp


### Specify Stan models

# measure time
compile_start <- Sys.time()

# FOR TESTING: model0 with target += normal_lpdf(theta[L] | 10, sqrt(tau2)); -> 10 instead of 0 to make the models more distinguishable

model0 <- 'data {
  int<lower = 1> N;               // Number of observations.
  int<lower = 1> L;               // Number of clusters.
  matrix[L, N] y;                 // Matrix of observations.
  real<lower=0> sigma_t;          // Higher order variance prior
  real<lower=0> sigma_s;          // Variance prior
  }
  parameters {
    real<lower=0> tau2; // Group-level variance
    vector[L] theta; // Participant effects
    real<lower=0> sigma2; // Unit-level variance
  }
  model {
    target += normal_lpdf(tau2 | 0, sigma_t) - normal_lcdf(0 | 0, sigma_t); // half-normal
    target += normal_lpdf(theta | 0, sqrt(tau2));
    target += normal_lpdf(sigma2 | 0, sigma_s) - normal_lcdf(0 | 0, sigma_s); // half-normal
    for (l in 1:L) {
        target += normal_lpdf(y[l] | theta[l], sqrt(sigma2));
    }
  }
  '
model1 <- 'data {
  int<lower = 1> N;               // Number of observations.
  int<lower = 1> L;               // Number of clusters.
  matrix[L, N] y;                 // Matrix of observations.
  real mu0;                       // higher order mean prior - mean
  real<lower=0> tau20;            // higher order mean prior - variance
  real<lower=0> sigma_t;          // Higher order variance prior
  real<lower=0> sigma_s;          // Variance prior
  }
  parameters {
    real mu;
    real<lower=0> tau2; // Group-level variance
    vector[L] theta; // Participant effects
    real<lower=0> sigma2; // Unit-level variance
  }
  model {
    target += normal_lpdf(mu | mu0, sqrt(tau20));
    target += normal_lpdf(tau2 | 0, sigma_t) - normal_lcdf(0 | 0, sigma_t); // half-normal
    target += normal_lpdf(theta | mu, sqrt(tau2));
    target += normal_lpdf(sigma2 | 0, sigma_s) - normal_lcdf(0 | 0, sigma_s); // half-normal  
    for (l in 1:L) {
        target += normal_lpdf(y[l] | theta[l], sqrt(sigma2));
    }
  }
  '

# Compile models
model0 <- stan_model(model_code = model0, model_name = "stanmodel")
model1 <- stan_model(model_code = model1, model_name = "stanmodel")

# Measure time
compile_end <- Sys.time()
compile_time <- difftime(compile_end, compile_start, unit="secs")
print(compile_time)


### Loop over test data sets

for (i in 1:dim(test_data)[1]){    
#for (i in 1:2){                       # FOR TESTING: 1:x
  
  # select data set
  test_data_single <- test_data[i,,,]
  
  # Measure stan start time
  stan_start <- Sys.time()
  

  ### Fit models
  
  # Prepare data & priors for sampling
  data_and_priors0 <- list(
    N = dim(test_data_single)[2],  # Number of observations.
    L = dim(test_data_single)[1],  # Number of clusters.
    y = test_data_single,          # Matrix of observations.
    sigma_t = 1,     
    sigma_s = 1
  )
  
  data_and_priors1 <- list(
    N = dim(test_data_single)[2],  # Number of observations.
    L = dim(test_data_single)[1],  # Number of clusters.
    y = test_data_single,          # Matrix of observations.
    mu0 = 0,                          # Difference to model0! 
    tau20 = 1,          
    sigma_t = 1,     
    sigma_s = 1
  )

  # Fit
  stanfit_model0 <- sampling(model0, data = data_and_priors0,
                        iter = 50000, warmup = 1000, chains = 4, cores = 4)
  
  stanfit_model1 <- sampling(model1, data = data_and_priors1,
                             iter = 50000, warmup = 1000, chains = 4, cores = 4)
  
  # Measure Stan end time
  stan_end <- Sys.time()
  
  # Save parameter estimates
  stanfits <- list(as.matrix(stanfit_model0),as.matrix(stanfit_model1))
  
  parameter_estimates <- save_parameter_estimates(i, stanfits, parameter_estimates, test_data_true_indices)
  
  ### Bridge Sampling
  
  # Measure Bridge Sampling start time
  bridge_start <- Sys.time()

  # Compute the (Log) Marginal Likelihoods
  m0.bridge <- bridge_sampler(stanfit_model0, silent = TRUE)
  m1.bridge <- bridge_sampler(stanfit_model1, silent = TRUE)

  # Compute Bayes factors
  bayes_factor_for_m1 <- bf(m1.bridge,m0.bridge) # Attention: order different in blog post (m0,m1)
  
  # Compute posterior model probabilities (assuming equal prior model probabilities)
  post_probs <- post_prob(m0.bridge, m1.bridge)
  
  # Compute selected model
  selected_model <- ifelse(post_probs[1] > 0.5, 0, 1)
  
  # Compute approximate percentage error of marginal likelihood estimates
  m0_bridge_error <- error_measures(m0.bridge)$percentage
  m1_bridge_error <- error_measures(m1.bridge)$percentage
  
  # Measure Bridge Sampling end time
  bridge_end <- Sys.time()
  
  # Calculate computation times
  stan_time <- difftime(stan_end, stan_start, unit="secs")
  bridge_time <- difftime(bridge_end, bridge_start, unit="secs")
  
  # Save comparison results
  comparison_results[i,"dataset"] <- i
  comparison_results[i,"true_model"] <- test_data_true_indices[i,2]
  comparison_results[i,"m0_prob"] <- post_probs[1]
  comparison_results[i,"m1_prob"] <- post_probs[2]   
  comparison_results[i,"selected_model"] <- selected_model
  comparison_results[i,"bayes_factor"] <- bayes_factor_for_m1$bf
  comparison_results[i,"m0_bridge_error"] <- m0_bridge_error
  comparison_results[i,"m1_bridge_error"] <- m1_bridge_error
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
exp_file_params <- sprintf("data/02_bridge_sampling_comparison/nested_models/%s_BF_BS_params", exp_time)
exp_file_comp <- sprintf("data/02_bridge_sampling_comparison/nested_models/%s_BF_BS", exp_time)

write.table(parameter_estimates, file=exp_file_params)
write.table(comparison_results, file=exp_file_comp)

### Load results of earlier experiments
old_params <- read.table("data/02_bridge_sampling_comparison/nested_models/2022_05_03_BF_BS_params")
old_comp <- read.table("data/02_bridge_sampling_comparison/nested_models/2022_05_03_BF_BS")



##### Diagnostics ######
# Inspect single stanfit objects

### Convential n_eff and Rhat statistics
summary(stanfit_model0)$summary
summary(stanfit_model1)$summary

#### Improved Rhat and ess statistics
# mu (only estimated in model1)
Rhat(extract_variable_matrix(stanfit_model1, "mu"))
ess_bulk(extract_variable_matrix(stanfit_model1, "mu"))
ess_tail(extract_variable_matrix(stanfit_model1, "mu"))

# tau2
Rhat(extract_variable_matrix(stanfit_model0, "tau2"))
ess_bulk(extract_variable_matrix(stanfit_model0, "tau2"))
ess_tail(extract_variable_matrix(stanfit_model0, "tau2"))

Rhat(extract_variable_matrix(stanfit_model1, "tau2"))
ess_bulk(extract_variable_matrix(stanfit_model1, "tau2"))
ess_tail(extract_variable_matrix(stanfit_model1, "tau2"))

# sigma2
Rhat(extract_variable_matrix(stanfit_model0, "sigma2"))
ess_bulk(extract_variable_matrix(stanfit_model0, "sigma2"))
ess_tail(extract_variable_matrix(stanfit_model0, "sigma2"))

Rhat(extract_variable_matrix(stanfit_model1, "sigma2"))
ess_bulk(extract_variable_matrix(stanfit_model1, "sigma2"))
ess_tail(extract_variable_matrix(stanfit_model1, "sigma2"))
