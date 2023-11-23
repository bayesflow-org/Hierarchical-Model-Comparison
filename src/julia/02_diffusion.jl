using Distributions
using AlphaStableDistributions

"Generates a single levy trial."
function generate_levy_trial(a_l::Float64, zr_l::Float64, v_l::Float64, t0_l::Float64, alpha_l::Float64, sz::Float64, sv::Float64, st::Float64, dt=0.001, tol=1e5)::Vector{Float64}
    
    # Declare variables
    n_steps = 0 # start at time 0
    rho = dt ^ (1.0/alpha_l) # Scaling factor -> how far into the trial is the current step

    # Include parameter variabilities (last terms serve to randomize direction of variability)
    zr_l = zr_l - 0.5*sz + sz * rand() # draw zr from uniform block between 0 and 1
    t0_l = t0_l - 0.5*st + st * rand() # draw t0 from uniform block between 0 and 1
    v_l = v_l + sv * rand(Normal()) # draw drift rate from Gaussian centered at v

    x = a_l * zr_l # scale trial starting point by respective threshold

    # Simulate single DM path
    while (x < a_l) && (x > 0) && (n_steps < tol) # as long as no decision boundary or the max. amount of steps is reached
        x += dt*v_l + rho * rand(AlphaStable(α=alpha_l, β=0.0, scale=1.0 / sqrt(2), location=0.0))
        n_steps += 1
    end
    
    return [dt*n_steps + t0_l, x > 0 ? 1.0 : 0.0] # reaction time, decision 
end

