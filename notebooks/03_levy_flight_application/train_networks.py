import os
import sys
import tensorflow as tf
import shutil
from functools import partial

# A convenient alternative to the split pretrain & finetune notebooks

# Silence tensorflow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Set paths
os.chdir(os.path.dirname(__file__))
sys.path.extend([
    os.path.abspath(os.path.join("../..")),
    os.path.abspath(os.path.join("../../../BayesFlow_dev/BayesFlow/"))
])

# Import from relative paths
from src.python.helpers import MaskingConfigurator
from src.python.settings import summary_meta_diffusion, probability_meta_diffusion
from src.python.training import load_training_data, setup_network
import bayesflow as bf

# Setup training
GOALS = ["pretrain", "finetune"]
LEARNING_RATES = {"pretrain": 0.0005, "finetune": 0.00005}
N_EPOCHS = {"pretrain": 20, "finetune": 30}
BATCH_SIZE = 32

# Start training loop
if __name__ == "__main__":
    for goal in GOALS:

        print(f'Starting to {goal}...')

        # Load data
        train_data = load_training_data(goal)
        val_data = load_training_data("validate")

        # Set up network
        tf.keras.backend.clear_session()
        summary_net, probability_net, amortizer = setup_network(
            summary_net_settings=summary_meta_diffusion,
            inference_net_settings=probability_meta_diffusion,
            loss_fun=partial(bf.losses.log_loss, label_smoothing=None)
        )

        # Set up trainer
        if goal == "finetune":
            shutil.copytree("checkpoints/pretrain", "checkpoints/finetune")
        checkpoint_path = f"checkpoints/{goal}"
        configurator = MaskingConfigurator() if goal == "finetune" else None
        trainer = bf.trainers.Trainer(
            amortizer=amortizer,
            configurator=configurator,
            checkpoint_path=checkpoint_path,
            default_lr=LEARNING_RATES[goal]
        )

        # Train
        losses = trainer.train_offline(
            simulations_dict=train_data,
            epochs=N_EPOCHS[goal],
            batch_size=BATCH_SIZE,
            validation_sims=val_data,
            **{"sim_dataset_args": {"batch_on_cpu": True}}
        )
