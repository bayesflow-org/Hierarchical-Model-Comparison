library(TreeBUGS)
library(LaplacesDemon)
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

### Goal: Compare TreeBUGS simulations to those by the Python simulator ###

readEQN(file = "2htm.txt", # relative or absolute path
        paramOrder = TRUE) # show parameter order

n_data_sets = 2500
n_clusters = 100
n_obs = 50

# Prepare matrices to store simulated data sets
mpt_data <- array(dim=c(n_data_sets, n_clusters, 2))

# Measure time
time_start <- Sys.time()

# Simulate data sets
for (d in 1:n_data_sets){
  
  # Sample hyperpriors as in Python simulator
  lambdas <- runif(n=2, min=0, max=10) # scaling params
  Q <- rinvwishart(nu=3, S=diag(2))
  sigma <- diag(lambdas) %*% Q %*% diag(lambdas) # Covariance matrix
  
  # Transform hyperpriors to TreeBUGS format
  sds <- sqrt(diag(sigma))
  corrs <- cov2cor(sigma) # same as cov2cor(Q) as corrs are determined by Q
  
  # Generate data
  genTrait <- genTraitMPT(
                N = n_clusters,                             # number of participants     
                numItems = c(Target=n_obs/2, Lure=n_obs/2),  # number of responses per tree
                eqnfile = "2htm.txt",                # path to MPT file
                mean = c(D=.5, g=.5),        # true group-level parameters
                sigma = sds,  # SD of latent (!) individual parameters
                rho = corrs)                       # correlation matrix
  
  # Save hit & false alarm rate per participant
  mpt_data[d,,1] <- genTrait$data[,"Hit"]/(n_obs/2)
  mpt_data[d,,2] <- genTrait$data[,"FA"]/(n_obs/2)
}

# Get time
time_end <- Sys.time()
time_total <- difftime(time_end, time_start, unit="secs")
print(time_total)

# Plot
hist(c(mpt_data[,,1]), main="Hit rate", xlab="") # Plot hit rate
hist(c(mpt_data[,,2]), main="False alarm rate", xlab="") # Plot false alarm rate

# Summary stats
mean(c(mpt_data[,,1])) # Mean hit rate
mean(c(mpt_data[,,2])) # Mean false alarm rate
