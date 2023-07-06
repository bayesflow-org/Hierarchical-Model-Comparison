# Network settings for validation experiment 1
summary_meta_validation_1 = {
    "level_1": {
        "dense_s1_args": dict(
                units=8, activation="elu", kernel_initializer="glorot_normal"
            ),
        "dense_s2_args": dict(
                units=32, activation="elu", kernel_initializer="glorot_normal"
            ),
        "dense_s3_args": dict(
                units=16, activation="elu", kernel_initializer="glorot_normal"
            )
    }, 
    "level_2": {
        "dense_s1_args": dict(
                units=32, activation="elu", kernel_initializer="glorot_normal"
            ),
        "dense_s2_args": dict(
                units=128, activation="elu", kernel_initializer="glorot_normal"
            ),
        "dense_s3_args": dict(
                units=64, activation="elu", kernel_initializer="glorot_normal"
            )
    }
}
        
probability_meta_validation_1 = {
    "dense_args": dict(units=64, activation="elu", kernel_initializer="glorot_normal")
}

# Network settings for validation experiment 2
summary_meta_validation_2 = {
    "level_1": {
        "dense_s1_args": dict(
                units=8, activation="elu", kernel_initializer="glorot_normal"
            ),
        "dense_s2_args": dict(
                units=32, activation="elu", kernel_initializer="glorot_normal"
            ),
        "dense_s3_args": dict(
                units=32, activation="elu", kernel_initializer="glorot_normal"
            )
    }, 
    "level_2": {
        "dense_s1_args": dict(
                units=64, activation="elu", kernel_initializer="glorot_normal"
            ),
        "dense_s2_args": dict(
                units=64, activation="elu", kernel_initializer="glorot_normal"
            ),
        "dense_s3_args": dict(
                units=64, activation="elu", kernel_initializer="glorot_normal"
            )
    }
}
        
probability_meta_validation_2 = {
    "dense_args": dict(units=128, activation="elu", kernel_initializer="glorot_normal")
}

# Network settings for levy flight application
summary_meta_diffusion = {
    "level_1": {
        "inv_inner": {
            "dense_inv_pre_pooling_args": dict(
                units=8, activation="elu", kernel_initializer="glorot_normal"
            ),
            "dense_inv_post_pooling_args": dict(
                units=8, activation="elu", kernel_initializer="glorot_normal"
            ),
            "n_dense_inv": 2,
        },
        "inv_outer": {
            "dense_inv_pre_pooling_args": dict(
                units=32, activation="elu", kernel_initializer="glorot_normal"
            ),
            "dense_inv_post_pooling_args": dict(
                units=32, activation="elu", kernel_initializer="glorot_normal"
            ),
            "n_dense_inv": 2,
        },
        "dense_equiv_args": dict(
            units=16, activation="elu", kernel_initializer="glorot_normal"
        ),
        "n_dense_equiv": 2,
        "n_equiv": 2,
    },
    "level_2": {
        "inv_inner": {
            "dense_inv_pre_pooling_args": dict(
                units=32, activation="elu", kernel_initializer="glorot_normal"
            ),
            "dense_inv_post_pooling_args": dict(
                units=32, activation="elu", kernel_initializer="glorot_normal"
            ),
            "n_dense_inv": 2,
        },
        "inv_outer": {
            "dense_inv_pre_pooling_args": dict(
                units=128, activation="elu", kernel_initializer="glorot_normal"
            ),
            "dense_inv_post_pooling_args": dict(
                units=128, activation="elu", kernel_initializer="glorot_normal"
            ),
            "n_dense_inv": 2,
        },
        "dense_equiv_args": dict(
            units=64, activation="elu", kernel_initializer="glorot_normal"
        ),
        "n_dense_equiv": 2,
        "n_equiv": 2,
    },
}
probability_meta_diffusion = {
    "dense_args": dict(units=64, activation="elu", kernel_initializer="glorot_normal"),
    "n_dense": 2,
    "n_models": 4,
}


# Plotting settings

plotting_settings = {
    "figsize": (5, 5),
    "colors_discrete": ("#440154FF", "#39568CFF", "#1F968BFF", "#73D055FF"),
    "alpha": 0.8,
    "fontsize_labels": 14,
    "fontsize_title": 16,
    "fontsize_legends": 12,
}


# BayesFlow default keys

DEFAULT_KEYS = {
    "prior_draws": "prior_draws",
    "obs_data": "obs_data",
    "sim_data": "sim_data",
    "batchable_context": "batchable_context",
    "non_batchable_context": "non_batchable_context",
    "prior_batchable_context": "prior_batchable_context",
    "prior_non_batchable_context": "prior_non_batchable_context",
    "prior_context": "prior_context",
    "hyper_prior_draws": "hyper_prior_draws",
    "shared_prior_draws": "shared_prior_draws",
    "local_prior_draws": "local_prior_draws",
    "sim_batchable_context": "sim_batchable_context",
    "sim_non_batchable_context": "sim_non_batchable_context",
    "summary_conditions": "summary_conditions",
    "direct_conditions": "direct_conditions",
    "parameters": "parameters",
    "hyper_parameters": "hyper_parameters",
    "shared_parameters": "shared_parameters",
    "local_parameters": "local_parameters",
    "observables": "observables",
    "targets": "targets",
    "conditions": "conditions",
    "posterior_inputs": "posterior_inputs",
    "likelihood_inputs": "likelihood_inputs",
    "model_outputs": "model_outputs",
    "model_indices": "model_indices",
}
