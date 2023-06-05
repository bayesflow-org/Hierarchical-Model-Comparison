"Generates reaction time datasets according to the four models in Wieschen, Voss and Radev (2020)."
function generate_levy_dataset_by_model(model::Int, n_clusters::Int64, n_trials::Int64)::Array{Float64, 3}

    data = fill(0.0, (n_clusters, n_trials, 3))
    hyperpriors = generate_levy_hyperpriors()

    for k in 1:n_clusters
        a_l, zr_l, v0_l, v1_l, t0_l, alpha_l, sz, sv, st = generate_levy_priors(hyperpriors...)
 
        if model == 1 || model == 3 # Gaussian noise
            alpha_l = 2.0
        end

        if model == 1 || model == 2 # Basic instead of full diffusion model
            sz = sv = st = 0.0 
        end

        data[k, :, :] = generate_levy_participant(n_trials, a_l, zr_l, v0_l, v1_l, t0_l, alpha_l, sz, sv, st)
    end

    return data
end


"Generates a batch of datasets simulated from a given model."
function generate_levy_batch(model::Int64, batch_size::Int64, n_clusters::Int64, n_trials::Int64)::Tuple{Vector{Int64}, Array{Float64, 4}}

    data = fill(0.0, (batch_size, n_clusters, n_trials, 3))
    index_list = fill(model, batch_size)

    Threads.@threads for b in 1:batch_size
        data[b, :, :, :] = generate_levy_dataset_by_model(model, n_clusters, n_trials)
    end

    return index_list, data
end