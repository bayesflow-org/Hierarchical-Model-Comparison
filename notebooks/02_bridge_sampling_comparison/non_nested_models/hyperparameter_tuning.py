import os, sys

# When on cloud (RStudio server)
#sys.path.append(os.path.abspath(os.path.join('../../..'))) # access sibling directories; as in .ipynb

# When on desktop (requires different paths than cloud)
sys.path.append(os.path.abspath(os.path.join('../Coding'))) # access sibling directories; different than .ipynb
sys.path.append("C:\\Users\\lasse\\Documents\\GitHub\\BayesFlow")

from src.python.settings import summary_meta_validation, evidence_meta_validation
from src.python.networks import HierarchicalInvariantNetwork, EvidentialNetwork
from src.python.models import HierarchicalSdtMptSimulator, MainSimulator
from src.python.losses import softmax_loss
from src.python.helpers import n_clust_obs_f_f, get_multiple_predictions, get_preds_and_bfs

import numpy as np
import pandas as pd
from tensorflow.keras.experimental import CosineDecay
from tensorflow.keras.optimizers import Adam

from functools import partial

from bayesflow.trainers import ModelComparisonTrainer
from bayesflow.amortizers import MultiModelAmortizer

# For testing: n_runs_per_setting = 1, epochs=2, iterations_per_epoch=2
# For running: n_runs_per_setting = 5, epochs=15, iterations_per_epoch=1000

#### Runs per setting combination 
n_runs_per_setting = 5

### Hyperparameter tuning params (cosine decay w/ restarts)
dropout_list = [False, True]
input_noise_list = [None, 0.05]

n_combinations = len(dropout_list)*len(input_noise_list)*n_runs_per_setting

### prepare static components
results = []

# Training steps
epochs=15 
iterations_per_epoch=1000

# Cosine decaying learning rate
initial_lr = 0.0005
decay_steps = epochs*iterations_per_epoch
alpha = 0
lr_schedule = CosineDecay(
    initial_lr, decay_steps, alpha=alpha
    )

# Sample size
n_clusters = 25
n_obs = 50

# Storage path
file_path = os.path.join(os.getcwd(),'notebooks\\02_bridge_sampling_comparison\\non_nested_models\\hyperparameter_tuning_results_dropout_input_noise')

### Run tuning
for dropout in dropout_list:
    for input_noise in input_noise_list:
        for run in range(n_runs_per_setting):

            if dropout == True:
                evidence_meta_validation.update({'dropout': True})
            if dropout == False:
                evidence_meta_validation.update({'dropout': False})
    
            # Initialize from scratch for each run
            summary_net = HierarchicalInvariantNetwork(summary_meta_validation)
            evidence_net = EvidentialNetwork(evidence_meta_validation)
    
            amortizer = MultiModelAmortizer(evidence_net, summary_net)
    
            simulator = MainSimulator(HierarchicalSdtMptSimulator())
    
            trainer = ModelComparisonTrainer(
                network=amortizer, 
                generative_model=simulator, 
                loss=partial(softmax_loss),
                optimizer=partial(Adam, lr_schedule),
                skip_checks=True,
                )
    
            # Train
            losses = trainer.train_online(
                epochs=1, iterations_per_epoch=1, batch_size=32, 
                n_obs=partial(n_clust_obs_f_f, n_clusters, n_obs), n_vars=2,
                add_noise_std=input_noise
                )
    
            # Get running loss of final epoch
            final_loss = np.mean(list(losses.values())[-1])
    
            # Store results
            results.append({
                'dropout': dropout,
                'input noise': input_noise,
                'final_loss': final_loss
            })
    
            # Print progress & secure progess
            if len(results) == round(n_combinations*0.25):
                print('25% done')
                pd.DataFrame(results).to_csv(file_path) # intermediate save
            if len(results) == round(n_combinations*0.50):
                print('50% done')
                pd.DataFrame(results).to_csv(file_path) # intermediate save
            if len(results) == round(n_combinations*0.75):
                print('75% done')
                pd.DataFrame(results).to_csv(file_path) # intermediate save

# Final saving of tuning results to csv
pd.DataFrame(results).to_csv(file_path)
