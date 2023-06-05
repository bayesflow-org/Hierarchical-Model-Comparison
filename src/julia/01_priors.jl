using Distributions
using StatsFuns

"Draws parameter values from the hyperprior distributions."
function generate_levy_hyperpriors()::Vector{Float64}

    # a (threshold separation)
    mu_a = rand(Normal(5.0, 1.0))
    sigma_a = rand(TruncatedNormal(0.4, 0.15, 0, Inf))
    
    # zr (relative starting point)
    mu_zr = rand(Normal(0.0, 0.25))
    sigma_zr = rand(TruncatedNormal(0, 0.05, 0, Inf))

    # v0 (drift rate for blue stimuli or non-word stimuli)
    mu_v0 = rand(Normal(5.0, 1.0))
    sigma_v0 = rand(TruncatedNormal(0.5, 0.25, 0, Inf))

    # v1 (drift rate for orange stimuli or word stimuli)
    mu_v1 = rand(Normal(5.0, 1.0))
    sigma_v1 = rand(TruncatedNormal(0.5, 0.25, 0, Inf))

    # t0 (non-decision time)
    mu_t0 = rand(Normal(5.0, 1.0))
    sigma_t0 = rand(TruncatedNormal(0.1, 0.05, 0, Inf))

    # alpha (stability parameter of noise distribution)
    mu_alpha = rand(Normal(1.65, 0.15))
    sigma_alpha = rand(TruncatedNormal(0.3, 0.1, 0, Inf))

    params = [mu_a, sigma_a,
                mu_zr, sigma_zr,
                mu_v0, sigma_v0,
                mu_v1, sigma_v1,
                mu_t0, sigma_t0,
                mu_alpha, sigma_alpha]

    return params
end


"Draws parameter values from the prior distributions given sampled hyperpriors."
function generate_levy_priors(mu_a::Float64, sigma_a::Float64, mu_zr::Float64, sigma_zr::Float64, mu_v0::Float64, sigma_v0::Float64,
                                mu_v1::Float64, sigma_v1::Float64,mu_t0::Float64, sigma_t0::Float64, mu_alpha::Float64, sigma_alpha::Float64)::Vector{Float64}

    # a (threshold separation)
    a_l = rand(Gamma(mu_a, sigma_a))
    
    # zr (relative starting point)
    zr_l = logistic.(rand(Normal(mu_zr, sigma_zr))) 
    # invlogit with nested normal to keep between 0 and 1

    # v0 (drift rate for blue stimuli or non-word stimuli)
    v0_l = -rand(Gamma(mu_v0, sigma_v0))

    # v1 (drift rate for orange stimuli or word stimuli)
    v1_l = rand(Gamma(mu_v1, sigma_v1))

    # t0 (non-decision time)
    t0_l = rand(Gamma(mu_t0, sigma_t0))

    # alpha (stability parameter of noise distribution)
    alpha_l = rand(TruncatedNormal(mu_alpha, sigma_alpha, 1, 2))
    #alpha_l = rand(Uniform(1,2)) # To check sensitivity against prior choice

    # intertrial variabilities
    sz = rand(Beta(1,3))
    sv = rand(TruncatedNormal(0, 2.0, 0, Inf))
    st = rand(TruncatedNormal(0, 0.3, 0, Inf))

    params = [a_l, zr_l, v0_l, v1_l, t0_l, alpha_l, sz, sv, st]

    return params
end