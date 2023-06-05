# Network settings for validation experiments
summary_meta_validation = {
    'level_1': {
        'inv_inner': {
            'dense_inv_pre_pooling_args'  : dict(units=8, activation='elu', kernel_initializer='glorot_normal'),
            'dense_inv_post_pooling_args' : dict(units=8, activation='elu', kernel_initializer='glorot_normal'),
            'n_dense_inv'                 : 2,
        },
        'inv_outer': {
            'dense_inv_pre_pooling_args'  : dict(units=32, activation='elu', kernel_initializer='glorot_normal'),
            'dense_inv_post_pooling_args' : dict(units=32, activation='elu', kernel_initializer='glorot_normal'),
            'n_dense_inv'                 : 2,
        },
        'dense_equiv_args': dict(units=16, activation='elu', kernel_initializer='glorot_normal'),
        'n_dense_equiv': 2,
        'n_equiv': 2,
    },
    'level_2': {
        'inv_inner': {
            'dense_inv_pre_pooling_args'  : dict(units=32, activation='elu', kernel_initializer='glorot_normal'),
            'dense_inv_post_pooling_args' : dict(units=32, activation='elu', kernel_initializer='glorot_normal'),
            'n_dense_inv'                 : 2,
        },
        'inv_outer': {
            'dense_inv_pre_pooling_args'  : dict(units=128, activation='elu', kernel_initializer='glorot_normal'),
            'dense_inv_post_pooling_args' : dict(units=128, activation='elu', kernel_initializer='glorot_normal'),
            'n_dense_inv'                 : 2,
        },
        'dense_equiv_args': dict(units=64, activation='elu', kernel_initializer='glorot_normal'),
        'n_dense_equiv': 2,
        'n_equiv': 2,
    },
}
probability_meta_validation = {
    'dense_args': dict(units=64, activation='elu', kernel_initializer='glorot_normal'),
    'n_dense': 2,
    'n_models': 2
}

# Network settings for levy flight application
summary_meta_diffusion = {
    'level_1': {
        'inv_inner': {
            'dense_inv_pre_pooling_args'  : dict(units=8, activation='elu', kernel_initializer='glorot_normal'),
            'dense_inv_post_pooling_args' : dict(units=8, activation='elu', kernel_initializer='glorot_normal'),
            'n_dense_inv'                 : 2,
        },
        'inv_outer': {
            'dense_inv_pre_pooling_args'  : dict(units=32, activation='elu', kernel_initializer='glorot_normal'),
            'dense_inv_post_pooling_args' : dict(units=32, activation='elu', kernel_initializer='glorot_normal'),
            'n_dense_inv'                 : 2,
        },
        'dense_equiv_args': dict(units=16, activation='elu', kernel_initializer='glorot_normal'),
        'n_dense_equiv': 2,
        'n_equiv': 2,
    },
    'level_2': {
        'inv_inner': {
            'dense_inv_pre_pooling_args'  : dict(units=32, activation='elu', kernel_initializer='glorot_normal'),
            'dense_inv_post_pooling_args' : dict(units=32, activation='elu', kernel_initializer='glorot_normal'),
            'n_dense_inv'                 : 2,
        },
        'inv_outer': {
            'dense_inv_pre_pooling_args'  : dict(units=128, activation='elu', kernel_initializer='glorot_normal'),
            'dense_inv_post_pooling_args' : dict(units=128, activation='elu', kernel_initializer='glorot_normal'),
            'n_dense_inv'                 : 2,
        },
        'dense_equiv_args': dict(units=64, activation='elu', kernel_initializer='glorot_normal'),
        'n_dense_equiv': 2,
        'n_equiv': 2,
    },
}
probability_meta_diffusion = {
    'dense_args': dict(units=64, activation='elu', kernel_initializer='glorot_normal'),
    'n_dense': 2,
    'n_models': 4
}


# Plotting settings

plotting_settings = {
    'figsize' : (5, 5),
    'colors_discrete' : ('#440154FF', '#39568CFF', '#1F968BFF', '#73D055FF'),
    'alpha' : 0.8,
    'fontsize_labels' : 14,
    'fontsize_title' : 16,
    'fontsize_legends' : 12
}