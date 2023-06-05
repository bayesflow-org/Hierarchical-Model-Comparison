setwd(dirname(rstudioapi::getSourceEditorContext()$path))


# Hierarchical SDT model (equal-variance model)
write("
data {
  int<lower = 1> M; // Number of clusters
  int<lower = 1> N; // Number of observations
  int<lower=0> N_old_new; // Number of old/new items, equal as 50/50 proportion of old and new items is given
  int X[M, N]; // 2D array of observations
}
  
transformed data {
  int X_h[M, N_old_new] = X[,1:N_old_new]; // Hits on old items
  int X_f[M, N_old_new] = X[,(N_old_new+1):N]; // False alarms on new items
}
  
parameters {
  // Hyperpriors
  real mu_h;
  real<lower = 0> sigma_h; 
  real mu_f;
  real<lower = 0> sigma_f; 
  
  // Group-level priors
  vector[M] h_m;
  vector[M] f_m;
}
  
transformed parameters {
  vector<lower = 0, upper = 1>[M] p_h_m;
  vector<lower = 0, upper = 1>[M] p_f_m;
  
  // Transform probit-transformed parameters to probabilities
  p_h_m = Phi(h_m);
  p_f_m = Phi(f_m);
}

model {
  // Hyperpriors
  target += normal_lpdf(mu_h | 1, 0.5);
  target += gamma_lpdf(sigma_h | 1, 1); // Careful: numpy uses scale, Stan inverse scale
  target += normal_lpdf(mu_f | -1, 0.5);
  target += gamma_lpdf(sigma_f | 1, 1); // Careful: numpy uses scale, Stan inverse scale

  // Group-level priors
  target += normal_lpdf(h_m | mu_h, sigma_h);
  target += normal_lpdf(f_m | mu_f, sigma_f);
  
  // Individual observations / likelihood
  for (m in 1:M) {
    target += bernoulli_lpmf(X_h[m] | p_h_m[m]);
    target += bernoulli_lpmf(X_f[m] | p_f_m[m]);
  }
}",
"sdt_model.stan")


# Test if model can be compiled
library(rstan)
stanc("sdt_model.stan")

# Hierarchical MPT model (latent-trait 2HTM)
write("
data {
  int<lower = 1> M; // Number of clusters
  int<lower = 1> N; // Number of observations
  int<lower=0> N_old_new; // Number of old/new items, equal as 50/50 proportion of old and new items is given
  int X[M, N]; // 2D array of observations
}
  
transformed data {
  int X_h[M, N_old_new] = X[,1:N_old_new]; // Hits on old items
  int X_f[M, N_old_new] = X[,(N_old_new+1):N]; // False alarms on new items
}
  
parameters {
  // Hyperpriors
  real mu_d;
  real mu_g;
  vector<lower = 0, upper = 2>[2] lambdas;
  cov_matrix[2] Q; // ensures symmetric and pos. definite outcome of inv-wishart sampling
  
  // Group-level priors
  matrix[2, M] z;
}
  
transformed parameters {
  cov_matrix[2] Sigma;
  cholesky_factor_cov[2] L; // Cholesky factor
  matrix[M, 2] params;
  vector<lower = 0, upper = 1>[M] p_d_m;
  vector<lower = 0, upper = 1>[M] p_g_m;
  vector<lower = 0, upper = 1>[M] p_h_m;
  vector<lower = 0, upper = 1>[M] p_f_m;
  
  // Calculate scaled inverse-wishart covariance matrix
  Sigma = (diag_matrix(lambdas) * Q) * diag_matrix(lambdas);
  
  // Apply cholesky decomposition to Sigma
  L = cholesky_decompose(Sigma);
  
  // Reparameterize multivariate normal
  params = (rep_matrix([mu_d, mu_g]', M) + L * z)';
  // rep_matrix matches the mu's to the [2, M] shape of z
  // transpose at the end to recieve more intuitive [M, 2] shape
  
  // Transform probit-transformed parameters to probabilities
  p_d_m = Phi(params[, 1]); // params[, 1] = d_m
  p_g_m = Phi(params[, 2]); // params[, 2] = g_m
  
  // Transform recognition/guess probs to hit/false alarm probs
  p_h_m = p_d_m + (1-p_d_m) .* p_g_m;
  p_f_m = (1-p_d_m) .* p_g_m;
}

model {
  // Hyperpriors
  target += normal_lpdf(mu_d | 0, 0.25);
  target += normal_lpdf(mu_g | 0, 0.25);
  target += uniform_lpdf(lambdas | 0.0, 2.0);
  target += inv_wishart_lpdf(Q| 3, diag_matrix(rep_vector(1.0, 2))); // diag_matrix(rep_vector(1, 2)) = stan version of R command diag(2)

  // Group-level priors
  for (m in 1:M) {
    target += std_normal_lpdf(z[, m]);
  }
  
  // Individual observations / likelihood
  for (m in 1:M) {
    target += bernoulli_lpmf(X_h[m] | p_h_m[m]);
    target += bernoulli_lpmf(X_f[m] | p_f_m[m]);
  }
}",
"mpt_model.stan")

# Test if model can be compiled
library(rstan)
stanc("mpt_model.stan")


