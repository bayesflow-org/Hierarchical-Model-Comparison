using Distributions

"Generates reaction time data from a partipant in a multi-stimulus task according to a diffusion model."
function generate_levy_participant(n_trials::Int64, a_l::Float64, zr_l::Float64, v0_l::Float64, v1_l::Float64, t0_l::Float64, 
                                    alpha_l::Float64, sz::Float64, sv::Float64, st::Float64)::Array{Float64, 2}

    data = fill(0.0, (n_trials, 3))
    stimulus_type = rand(Bernoulli(0.5), n_trials) # sample stimulus type
    data[:, 1] = stimulus_type

    for n in 1:n_trials

        if stimulus_type[n] == false # blue / non-word
            data[n, 2:3] = generate_levy_trial(a_l, zr_l, v0_l, t0_l, alpha_l, sz, sv, st)
        else # orange / word
            data[n, 2:3] = generate_levy_trial(a_l, zr_l, v1_l, t0_l, alpha_l, sz, sv, st)
        end
        
    end
    
    return data # stimulus_type, rt, decision
end
