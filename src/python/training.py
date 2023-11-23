import os
import sys
import pickle

sys.path.append(os.path.abspath(os.path.join('../../../BayesFlow_dev/BayesFlow/')))
import bayesflow as bf


# Levy flight application
def load_training_data(goal):
    """ Loads data for the training process according to goal. """
    data_path = os.path.abspath(f"../../data/03_levy_flight_application/simulated_data/{goal}.pkl")
    with open(data_path, "rb") as file:
        return pickle.load(file)


def setup_network(summary_net_settings, inference_net_settings, loss_fun):
    """ Sets up a hierarchical model comparison network with standardized settings. """
    summary_net = bf.summary_networks.HierarchicalNetwork([
        bf.networks.DeepSet(
            summary_dim=summary_net_settings['level_1']['summary_dim'],
            dense_s1_args=summary_net_settings['level_1']['dense_s1_args'],
            dense_s2_args=summary_net_settings['level_1']['dense_s2_args'],
            dense_s3_args=summary_net_settings['level_1']['dense_s3_args']
        ),
        bf.networks.DeepSet(
            summary_dim=summary_net_settings['level_2']['summary_dim'],
            dense_s1_args=summary_net_settings['level_2']['dense_s1_args'],
            dense_s2_args=summary_net_settings['level_2']['dense_s2_args'],
            dense_s3_args=summary_net_settings['level_2']['dense_s3_args']
        )])

    probability_net = bf.inference_networks.PMPNetwork(
        num_models=inference_net_settings['num_models'],
        dense_args=inference_net_settings['dense_args'],
        num_dense=inference_net_settings['num_dense'],
        dropout=inference_net_settings['dropout']
    )

    amortizer = bf.amortizers.AmortizedModelComparison(
        probability_net,
        summary_net,
        loss_fun=loss_fun
    )

    return summary_net, probability_net, amortizer
