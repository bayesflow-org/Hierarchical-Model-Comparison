import numpy as np
np.set_printoptions(suppress=True)
from tensorflow.keras.utils import to_categorical
from scipy import stats

import tensorflow as tf


class HierarchicalNormalSimulator:
    
    def __init__(self):    
        pass
    
    def draw_from_prior(self, model_index, n_clusters, mu0, tau20, sigma_t, sigma_s):
        """
        Draws parameter values from the specified prior distributions of the 
        hyperprior and the conditional prior.
        ----------
        
        Arguments:
        model_index : int -- index of the model to be simulated from
        n_clusters  : int -- number of higher order clusters that the observations are nested in
        mu0         : float -- higher order mean prior - mean
        tau20       : float -- higher order mean prior - variance
        sigma_t     : float -- higher order variance prior
        sigma_s     : float -- variance prior
        """
        
        if model_index == 0: 
            mu = 0
        if model_index == 1:
            mu = np.random.normal(loc=mu0, scale=np.sqrt(tau20))
            
        tau2 = stats.halfnorm.rvs(scale=sigma_t)
        sigma2 = stats.halfnorm.rvs(scale=sigma_s)
        theta = np.random.normal(loc=mu, scale=np.sqrt(tau2), size=n_clusters)
        return theta, sigma2
    
    def gen_from_likelihood(self, theta, sigma2, n_obs):
        """
        Generates a single hierarchical data set from the sampled parameter values.
        ----------
        
        Arguments: 
        params : list -- parameters sampled from prior 
        n_obs  : int -- number of observations per cluster
        """
        
        X = np.random.normal(loc=theta, scale=np.sqrt(sigma2), size=(n_obs, theta.shape[0])).T 
        return X
    
    def generate_single(self, model_index, n_clusters, n_obs, mu0=0, tau20=1, sigma_t=1, sigma_s=1):
        """
        Generates a single hierarchical data set utilizing the draw_from_prior and gen_from_likelihood functions.
        ----------
        
        Arguments:
        model_index : int -- index of the model to be simulated from
        n_clusters  : int -- number of higher order clusters that the observations are nested in
        n_obs       : int -- number of observations per cluster
        mu0         : float -- higher order mean prior - mean
        tau20       : float -- higher order mean prior - variance
        sigma_t     : float -- higher order variance prior
        sigma_s     : float -- variance prior
        --------
        
        Returns:
        numpy array of shape (n_clusters, n_obs, n_variables) - contains the simulated hierarchical data sets
        """
        theta, sigma2 = self.draw_from_prior(model_index, n_clusters, mu0, tau20, sigma_t, sigma_s)
        x_generated = self.gen_from_likelihood(theta, sigma2, n_obs)
        return x_generated[...,np.newaxis]
        
    def simulate(self, batch_size, n_clusters, n_obs, mu0=0, tau20=1, sigma_t=1, sigma_s=1):
        """
        Simulates multiple hierarchical data sets. Useful for single usage and debugging (both without the MainSimulator).
        ----------
        
        Arguments:
        batch_size  : int -- number of batches to be generated
        n_models    : int -- number of models to be simulated from
        n_clusters  : int -- number of higher order clusters that the observations are nested in
        n_obs       : int -- number of observations per cluster
        n_variables : int -- number of variables in the simulated data sets 
        mu0         : float -- higher order mean prior - mean
        tau20       : float -- higher order mean prior - variance
        sigma_t     : float -- higher order variance prior
        sigma_s     : float -- variance prior
        --------
        
        Returns:
        numpy array of shape (batch_size * n_models, n_clusters, n_obs, n_variables) - contains the simulated hierarchical data sets
        """
        
        X = []
        for b in range(batch_size):
            prior_sample = self.draw_from_prior(n_clusters, mu0, tau20, sigma_t, sigma_s)
            x_generated = self.gen_from_likelihood(prior_sample, n_obs)
            X.append(x_generated)
        return np.array(X)[...,np.newaxis]


class HierarchicalSdtMptSimulator:

    def __init__(self):
        pass

    def draw_from_prior(self, model_index, n_clusters):
        """ Draws parameter values from the specified prior distributions of the hyperprior and the conditional prior.
            Also computes hit and false alarm probabilities for each participant (to be compatible with the MainSimulator class).

        Parameters
        ----------
        model_index : int
            Index of the model to be simulated from.
        n_clusters : int
            Number of higher order clusters that the observations are nested in.

        Returns
        -------
        p_h_m : np.array
            Hit probability per cluster.
        p_h_m : np.array
            False alarm probability per cluster.
        """

        RNG = np.random.default_rng()
        
        if model_index == 0: # SDT model

            # Hyperpriors
            mu_h = RNG.normal(1, 0.5)
            sigma_h = RNG.gamma(1, 1)
            mu_f = RNG.normal(-1, 0.5)
            sigma_f = RNG.gamma(1, 1)

            # Group-level priors
            h_m = RNG.normal(loc=mu_h, scale=sigma_h, size=n_clusters)
            f_m = RNG.normal(loc=mu_f, scale=sigma_f, size=n_clusters)

            # Transform probit-transformed parameters to probabilities
            p_h_m = stats.norm.cdf(h_m) 
            p_f_m = stats.norm.cdf(f_m)

        if model_index == 1: # MPT model

            # Hyperpriors
            mu_d = RNG.normal(0, 0.25) 
            mu_g = RNG.normal(0, 0.25)
            lambdas = RNG.uniform(0, 2, size=2) # upper limit determines scaling of correlation matrix (low = low sds of random effects)
            Q = stats.invwishart.rvs(df=3, scale=np.identity(2)) # df determines correlations between d and g (low = higher correlations, 3 = uniform)
            sigma = np.matmul(np.matmul(np.diag(lambdas), Q), np.diag(lambdas))

            # Group-level priors
            params = RNG.multivariate_normal([mu_d, mu_g], sigma, size=n_clusters)
            d_m = params[:, 0]
            g_m = params[:, 1]

            # Transform probit-transformed parameters to probabilities
            p_d_m = stats.norm.cdf(d_m) 
            p_g_m = stats.norm.cdf(g_m) 
            
            # Transform recognition / guess probs to hit / false alarm probs
            p_h_m = p_d_m + (1-p_d_m)*p_g_m
            p_f_m = (1-p_d_m)*p_g_m
        
        return p_h_m, p_f_m

    def generate_from_likelihood(self, p_h_m, p_f_m, n_clusters, n_obs):
        """ Generates a single hierarchical data set from the sampled parameter values.

        Parameters
        ----------
        p_h_m : np.array
            Hit probability per cluster.
        p_h_m : np.array
            False alarm probability per cluster.
        n_clusters : int
            Number of higher order clusters that the observations are nested in.
        n_obs : int
            Number of observations per cluster.

        Returns
        -------
        X : np.array
            Generated data set with shape (n_clusters, n_obs, 2).
            Contains 2 binary variables with stimulus type and response (for both applies: 0="new" / 1="old").
        """
        
        RNG = np.random.default_rng()

        # Determine amount of signal (old item) and noise (new item) trials
        assert n_obs%2 == 0, "n_obs has to be dividable by 2."
        n_trials_per_cat = int(n_obs/2)

        # Create stimulus types (0="new" / 1="old")
        stim_cluster = np.repeat([[1,0]], repeats=n_trials_per_cat, axis=1) # For 1 participant
        stim_data_set = np.repeat(stim_cluster, repeats=n_clusters, axis=0) # For 1 data set

        # Create individual responses (0="new" / 1="old")
        X_h = RNG.binomial(n=1, p=p_h_m, size=(n_trials_per_cat, n_clusters)).T # Old items
        X_f = RNG.binomial(n=1, p=p_f_m, size=(n_trials_per_cat, n_clusters)).T # New items
        X_responses = np.concatenate((X_h, X_f), axis=1)

        # Create final data set
        X = np.stack((stim_data_set, X_responses), axis=2)

        return X

    def generate_single(self, model_index, n_clusters, n_obs):
        """Generates a single hierarchical data set utilizing the draw_from_prior and gen_from_likelihood functions.

        Parameters
        ----------
        model_index : int
            Index of the model to be simulated from.
        n_clusters : int
            Number of higher order clusters that the observations are nested in.
        n_obs : int
            Number of observations per cluster.

        Returns
        -------
        X : np.array
            Generated data set with shape (n_clusters, n_obs, 2).
            Contains 2 binary variables with stimulus type and response (for both applies: 0="new" / 1="old").
        """

        p_h_m, p_f_m = self.draw_from_prior(model_index, n_clusters)
        X = self.generate_from_likelihood(p_h_m, p_f_m, n_clusters, n_obs)

        return X


class MainSimulator:
    
    def __init__(self, simulator):
        
        self.simulator = simulator
    
    def draw_from_model_prior(self, batch_size, n_models, model_prior):
        """
        Creates the sequence of models to be simulated from in the batch.
        ----------
        
        Arguments:
        batch_size     : int -- number of batches to be generated
        n_models       : int -- number of models to be simulated from
        model_prior    : list -- prior model probabilities
        --------
        
        Returns:
        array of shape (batch_size) - array of indices corresponding to the sampled model from p(M).
        """
        
        # create base list of model indices
        model_base_indices = [*range(n_models)]
        
        # uniform prior over model probabilities if no model prior given
        if model_prior == None:
            model_prior = [1/n_models] * n_models
        
        # generate sampling list of model indeces
        model_indices = np.random.choice(model_base_indices, size=batch_size, p=model_prior)
        return model_indices
    
    def simulate(self, batch_size, n_obs, n_vars, n_models, model_prior):
        """
        Simulates a batch of hierarchical data sets.
        ----------
        
        Arguments:
        batch_size     : int -- number of data sets to be generated per batch
        n_obs          : function -- generates the number of clusters and observations in the batch
        n_vars         : int -- number of variables per data set
        n_models       : int -- number of models to be simulated from
        model_prior    : list -- prior model probabilities
        --------
        
        Returns:
        dict of {'X' : array of shape (batch_size, n_clusters, n_obs, n_variables),  
                 'm' : array of shape (batch_size)}
        """
        # Draw K and N (drawn values apply for all data sets in the batch)
        K, N = n_obs
        
        # Draw sampling list of model indices
        model_indices = self.draw_from_model_prior(batch_size, n_models, model_prior)
        
        # Prepare an array to hold simulations
        X_gen = np.zeros((batch_size, K, N, n_vars), dtype=np.float32)

        for b in range(batch_size):
            X_gen[b] = self.simulator.generate_single(model_indices[b], K, N)
               
        return to_categorical(model_indices), None, X_gen
    
    def __call__(self, batch_size, n_obs, n_vars=1, n_models=2, model_prior=None):
        return self.simulate(batch_size, n_obs, n_vars, n_models, model_prior)