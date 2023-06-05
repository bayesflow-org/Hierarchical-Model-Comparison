using Distributions
using AlphaStableDistributions
using Statistics
using Plots
using StatsPlots
using StatsFuns

include("01_priors.jl")
include("02_diffusion.jl")
include("03_experiment.jl")
include("04_datasets.jl")

# Single trial generation
generate_levy_trial(1.0, 0.5, 2.0, 0.5, 2.0, 0.3, 0.7, 0.2);

# Single participant generation
generate_levy_participant(900, 1.0, 0.5, -2.0, 2.0, 0.5, 2.0, 0.3, 0.7, 0.2);

# Single participant in two conditions generation
generate_levy_conditions(900, 1.0, 0.5, -2.0, 2.0, 0.5, 2.0, 0.3, 0.7, 0.2);

# Hyperprior generation
generate_levy_hyperpriors();

# (lower level) Prior generation
h_test = generate_levy_hyperpriors()
generate_levy_priors(h_test...);

# Dataset generation by model
generate_levy_dataset_by_model(1, 40, 900);
generate_levy_dataset_by_model(2, 40, 900);
generate_levy_dataset_by_model(3, 40, 900);
generate_levy_dataset_by_model(4, 40, 900);

# Batch of datasets generation
generate_levy_batch(1, 2, 40, 900);
generate_levy_batch(2, 2, 40, 900);
generate_levy_batch(3, 2, 40, 900);
generate_levy_batch(4, 2, 40, 900);

println("Done!");