import os, sys

# When on cloud (RStudio server)
# sys.path.append(os.path.abspath(os.path.join('../../..'))) # access sibling directories; as in .ipynb

# When on desktop (requires different paths than cloud)
sys.path.append(
    os.path.abspath(os.path.join("../../.."))
)  # access sibling directories; different than .ipynb
sys.path.append(os.path.abspath(os.path.join("../../../../BayesFlow_dev/BayesFlow/")))

from src.python.models import HierarchicalSdtMptSimulator

import numpy as np
import bayesflow as bf
import pandas as pd
import tensorflow as tf
from time import perf_counter

from functools import partial


# For testing: n_runs_per_setting = 1, epochs=2, iterations_per_epoch=2
# For running: n_runs_per_setting = 5, epochs=15, iterations_per_epoch=1000

#### Runs per setting combination
n_runs_per_setting = 2  # 5

### Hyperparameter tuning params (cosine decay w/ restarts)
bigger_network_128_units = [False, True]
smaller_lr_5e_5 = [False, True]

n_combinations = (
    len(bigger_network_128_units) * len(smaller_lr_5e_5) * n_runs_per_setting
)

### prepare static components
results = []

# Sample size
n_clusters = 25
n_obs = 50

# Generative models
sdtmpt_model = HierarchicalSdtMptSimulator()

sdt_simulator = partial(
    sdtmpt_model.generate_batch,
    model_index=0,
    n_clusters=n_clusters,
    n_obs=n_obs,
    n_vars=2,
)
mpt_simulator = partial(
    sdtmpt_model.generate_batch,
    model_index=1,
    n_clusters=n_clusters,
    n_obs=n_obs,
    n_vars=2,
)
meta_model = bf.simulation.MultiGenerativeModel([sdt_simulator, mpt_simulator])

# Training steps
epochs = 3  # 20
iterations_per_epoch = 10  # 00

# Storage path
file_path = os.path.join(os.getcwd(), "hyperparameter_tuning_results_size_lr")

### Run tuning
for bigger_size in bigger_network_128_units:
    for smaller_lr in smaller_lr_5e_5:
        for run in range(n_runs_per_setting):
            if bigger_network_128_units == True:
                config = {
                    "units": 128,
                    "activation": "relu",
                    "kernel_initializer": "glorot_uniform",
                }
                summary_net = bf.summary_networks.HierarchicalNetwork(
                    [
                        bf.networks.DeepSet(
                            dense_s1_args=config,
                            dense_s2_args=config,
                            dense_s3_args=config,
                        ),
                        bf.networks.DeepSet(
                            dense_s1_args=config,
                            dense_s2_args=config,
                            dense_s3_args=config,
                        ),
                    ]
                )
            else:
                summary_net = bf.summary_networks.HierarchicalNetwork(
                    [bf.networks.DeepSet(), bf.networks.DeepSet()]
                )

            # Initialize from scratch for each run
            probability_net = bf.inference_networks.PMPNetwork(
                num_models=2, dropout=False
            )
            amortizer = bf.amortizers.AmortizedModelComparison(
                probability_net, summary_net
            )

            if smaller_lr:
                initial_lr = 0.00005  # smaller than default!
                schedule = tf.keras.optimizers.schedules.CosineDecay(
                    initial_lr, iterations_per_epoch * epochs, name="lr_decay"
                )
                optimizer = tf.keras.optimizers.Adam(schedule)
                trainer = bf.trainers.Trainer(
                    amortizer=amortizer,
                    generative_model=meta_model,
                    optimizer=optimizer,
                )

            else:
                trainer = bf.trainers.Trainer(
                    amortizer=amortizer, generative_model=meta_model
                )

            # Train
            training_time_start = perf_counter()
            losses = trainer.train_online(
                epochs=epochs, iterations_per_epoch=iterations_per_epoch, batch_size=32
            )
            training_time_stop = perf_counter()

            # Get running loss of final epoch and training time
            final_loss = np.mean(losses[-iterations_per_epoch:])
            training_time = training_time_stop - training_time_start

            # Store results
            results.append(
                {
                    "bigger_size": bigger_size,
                    "smaller_lr": smaller_lr,
                    "final_loss": final_loss,
                    "training_time": training_time,
                }
            )

            # Print progress & secure progess
            if len(results) == round(n_combinations * 0.25):
                print("25% done")
                pd.DataFrame(results).to_csv(file_path)  # intermediate save
            if len(results) == round(n_combinations * 0.50):
                print("50% done")
                pd.DataFrame(results).to_csv(file_path)  # intermediate save
            if len(results) == round(n_combinations * 0.75):
                print("75% done")
                pd.DataFrame(results).to_csv(file_path)  # intermediate save

# Final saving of tuning results to csv
pd.DataFrame(results).to_csv(file_path)
