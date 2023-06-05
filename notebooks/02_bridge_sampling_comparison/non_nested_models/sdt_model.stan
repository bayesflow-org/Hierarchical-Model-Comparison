
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
}
