import os, sys
sys.path.append(os.path.abspath(os.path.join('../../../BayesFlow_dev/BayesFlow/')))
import bayesflow as bf
import numpy as np
import pandas as pd
import tensorflow as tf
import copy
from scipy.stats import truncnorm
from tensorflow.keras.utils import to_categorical
from time import perf_counter
from tqdm.notebook import tqdm
from functools import partial

from sklearn.utils import column_or_1d, assert_all_finite, check_consistent_length
from sklearn.metrics import accuracy_score
from sklearn.metrics._base import _check_pos_label_consistency


# Calibration
# Variable sample sizes

def variable_n_obs(n_obs_min, n_obs_max):
    n_obs = np.random.randint(n_obs_min, n_obs_max)
    return {'n_obs' : n_obs}

def variable_sizes(n_clusters_min, n_clusters_max, n_obs_min, n_obs_max):
    n_clusters = np.random.randint(n_clusters_min, n_clusters_max)
    n_obs = np.random.randint(n_obs_min, n_obs_max)
    return {'n_clusters' : n_clusters, 'n_obs' : n_obs}

# Get calibration curve and ECE
def calibration_curve_with_ece(
    y_true, y_prob, pos_label=None, n_bins=15, strategy="uniform"
):
    """
    [sklearn.calibration.calibration_curve source code supplemented with an ECE calculation
     proposed by chrisyeh96 in issue #18268 of the scikit-learn github repository.]
    Compute true and predicted probabilities for a calibration curve.
    The method assumes the inputs come from a binary classifier, and
    discretize the [0, 1] interval into bins.
    Calibration curves may also be referred to as reliability diagrams.
    Computes ECE according to Naeini et al. (2015) with prob_true
    and NOT according to Guo et al. (2017) which use accuracy instead.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True targets.
    y_prob : array-like of shape (n_samples,)
        Probabilities of the positive class.
    pos_label : int or str, default=None
        The label of the positive class.
        .. versionadded:: 1.1
    n_bins : int, default=5
        Number of bins to discretize the [0, 1] interval. A bigger number
        requires more data. Bins with no samples (i.e. without
        corresponding values in `y_prob`) will not be returned, thus the
        returned arrays may have less than `n_bins` values.
    strategy : {'uniform', 'quantile'}, default='uniform'
        Strategy used to define the widths of the bins.
        uniform
            The bins have identical widths.
        quantile
            The bins have the same number of samples and depend on `y_prob`.
    Returns
    -------
    prob_true : ndarray of shape (n_bins,) or smaller
        The proportion of samples whose class is the positive class, in each
        bin (fraction of positives).
    prob_pred : ndarray of shape (n_bins,) or smaller
        The mean predicted probability in each bin.
    ece : float
        The ECE over all bins.
    References
    ----------
    Alexandru Niculescu-Mizil and Rich Caruana (2005) Predicting Good
    Probabilities With Supervised Learning, in Proceedings of the 22nd
    International Conference on Machine Learning (ICML).
    See section 4 (Qualitative Analysis of Predictions).
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.calibration import calibration_curve
    >>> y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    >>> y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.65, 0.7, 0.8, 0.9,  1.])
    >>> prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=3)
    >>> prob_true
    array([0. , 0.5, 1. ])
    >>> prob_pred
    array([0.2  , 0.525, 0.85 ])
    """
    y_true = column_or_1d(y_true)
    y_prob = column_or_1d(y_prob)
    check_consistent_length(y_true, y_prob)
    pos_label = _check_pos_label_consistency(pos_label, y_true)

    if y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError("y_prob has values outside [0, 1].")

    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError(
            f"Only binary classification is supported. Provided labels {labels}."
        )
    y_true = y_true == pos_label

    if strategy == "quantile":  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
    elif strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(
            "Invalid entry to 'strategy' input. Strategy "
            "must be either 'quantile' or 'uniform'."
        )

    binids = np.searchsorted(bins[1:-1], y_prob)

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]

    # Calculate ECE as proposed by chrisyeh96 (https://github.com/scikit-learn/scikit-learn/issues/18268)
    ece = np.sum(np.abs(prob_true - prob_pred) * (bin_total[nonzero] / len(y_true)))

    return prob_true, prob_pred, ece


# Get multiple predictions to plot with uncertainty.
def get_repeated_predictions(
    probability_net,
    summary_net,
    trainer,
    simulator,
    n_models,
    n_data_sets=5000,
    n_repetitions=10,
):
    """Gets repeated predictions from the trained hierarchical network via repeated simulation and prediction.
    Recommended for online learning settings with fast data set simulation.

    Parameters
    ----------
    probability_net : ModelProbabilityNetwork
        Trained model probability network.
    summary_net : HierarchicalInvariantNetwork
        Trained summary network.
    simulator : MainSimulator
        A generative model
    n_models : int
        Number of compared models
    n_data_sets : int
        Number of data sets simulated in each repetition.
    n_repetitions : int
        Number of times the simulation and prediction loop is repeated.

    Returns
    -------
    m_true, m_soft : np.arrays of shape (n_repetitions, n_data_sets, n_models)
        True model indices and corresponding predictions for each model.
    """

    m_soft = np.zeros((n_repetitions, n_data_sets, n_models))
    m_true = np.zeros((n_repetitions, n_data_sets, n_models))

    for i in range(n_repetitions):
        data = trainer.configurator(simulator(n_data_sets))

        m_soft[i, :, :] = np.concatenate(
            [
                probability_net.posterior_probs(summary_net(x_chunk))
                for x_chunk in tf.split(data["summary_conditions"], 20)
            ]
        )
        m_true[i, :, :] = data["model_indices"]

    return m_true, m_soft


def get_bootstrapped_predictions(
    probability_net,
    summary_net,
    simulated_data,
    simulated_indices,
    n_models,
    n_bootstrap=100,
):
    """Gets bootstrapped predictions from the trained hierarchical network via bootstrapping of the given batch of data sets.
    Recommended for offline learning settings or online learning settings with slow data set simulation.

    Parameters
    ----------
    probability_net : ModelProbabilityNetwork
        Trained model probability network.
    summary_net : HierarchicalInvariantNetwork
        Trained summary network.
    simulated_data : np.array
        A batch of simulated data sets
    simulated_indices : np.array
        A batch of simulated indices marking which model each data set in simulated_data was generated from.
    n_models : int
        Number of compared models
    n_bootstrap : int
        Number of bootstrap samples from to simulated_data.

    Returns
    -------
    m_true, m_soft : np.arrays of shape (n_repetitions, n_data_sets, n_models)
        True model indices and corresponding predictions for each model.
    """

    n_data_sets = simulated_indices.shape[0]

    m_soft = np.zeros((n_bootstrap, n_data_sets, n_models))
    m_true = np.zeros((n_bootstrap, n_data_sets, n_models))

    for i in range(n_bootstrap):
        b_idx = np.random.choice(np.arange(n_data_sets), size=n_data_sets, replace=True)
        m_soft[i, :, :] = tf.concat(
            [
                probability_net.predict(summary_net(x_chunk))["m_probs"]
                for x_chunk in tf.split(simulated_data[b_idx], 20)
            ],
            axis=0,
        )
        m_true[i, :, :] = simulated_indices[b_idx]

    return m_true, m_soft


# Calibration: Plotting for training with variable numbers of clusters and variable number of observations
def compute_eces_variable(
    probability_net,
    summary_net,
    trainer,
    m1_simulator,
    m2_simulator,
    n_val_per_setting,
    n_clust_min,
    n_clust_max,
    n_obs_min,
    n_obs_max,
    n_cal_bins=15,
    add_accuracy_sbc=False,
):
    """
    Simulates validation data per setting and computes the expected calibration error of the model.
    --------

    Returns:
    2 lists of shape((n_clust_max+1 - n_clust_min)*(n_obs_max+1 - n_obs_min))
    - containing the mean (1st list) / sd (2nd list) eces of all possible combinations on L and N.
    """


    # Create lists
    eces = []
    if add_accuracy_sbc:
        accuracies = []
        sbcs = []

    with tqdm(
        total=(n_clust_max + 1 - n_clust_min), desc="Loop through clusters progress"
    ) as p_bar:
        with tqdm(
            total=(n_obs_max + 1 - n_obs_min),
            desc="Loop through nested observations progress",
        ) as p_bar_within:
            for l in range(n_clust_min, n_clust_max + 1):  # Loop through clusters
                p_bar_within.reset(
                    (n_obs_max + 1 - n_obs_min)
                )  # reuse 2nd bar so that screen doesn't explode
                for n in range(
                    n_obs_min, n_obs_max + 1
                ):  # Loop through nested observations
                    
                    # Generate data
                    meta_simulator = bf.simulation.MultiGenerativeModel(
                        [m1_simulator, m2_simulator], 
                        shared_context_gen=partial(variable_sizes, l, l+1, n, n+1)
                        )
                    data = trainer.configurator(meta_simulator(n_val_per_setting))

                    # Predict model probabilities
                    split_direct_conditions = tf.split(data["direct_conditions"], 20)
                    preds = []
                    for i, x_chunk in enumerate(tf.split(data["summary_conditions"], 20)):
                        embedding = summary_net(x_chunk)
                        inference_net_input = np.concatenate([embedding, split_direct_conditions[i]], axis=1)
                        preds.append(probability_net.posterior_probs(inference_net_input)[:, 1])
                    m_soft = np.concatenate(preds)
                    m_hard = (m_soft > 0.5).astype(np.int32)
                    m_true = data["model_indices"][:, 1]

                    # Compute metrics
                    prob_true, prob_pred, ece = calibration_curve_with_ece(
                        m_true, m_soft, n_bins=n_cal_bins
                    )
                    eces.append(ece)

                    if add_accuracy_sbc:
                        accuracy = accuracy_score(m_true, m_hard)
                        accuracies.append(accuracy)

                        sbc = np.mean(0.5 - np.mean(m_soft))
                        sbcs.append(sbc)

                    # Update inner progress bar
                    p_bar_within.set_postfix_str(
                        "Cluster {0}, Observation {1}".format(l, n + 1)
                    )
                    p_bar_within.update()

                # Refresh inner + update outer progress bar
                p_bar_within.refresh()
                p_bar.set_postfix_str("Finished clusters: {}".format(l))
                p_bar.update()

    if add_accuracy_sbc:
        return eces, accuracies, sbcs

    return eces


# Bridge sampling comparison: data transformations


def get_preds_and_bfs(
    probability_net, summary_net, data, training_time_start, training_time_stop, losses, variable=False
):
    """
    Writes model predictions and resulting Bayes Factors for a given
    array of datasets into a pandas DataFrame.
    final_epoch_loss assumes that each epoch contains 1000 training steps.
    """

    n_datasets = data["X"].shape[0]
    dataset = np.arange(1, n_datasets + 1)
    true_model = data["m"][:, 1]

    # Predict
    inference_time_start = perf_counter()
    if variable:
        n_clusters_rep = np.sqrt(data["X"].shape[1] * np.ones((n_datasets, 1)))
        n_obs_rep = np.sqrt(data["X"].shape[2] * np.ones((n_datasets, 1)))
        n_rep = np.concatenate((n_clusters_rep, n_obs_rep), axis=-1)
        split_direct_conditions = tf.split(n_rep, 20)
        preds = []
        for i, x_chunk in enumerate(tf.split(data["X"], 20)):
            embedding = summary_net(x_chunk)
            inference_net_input = np.concatenate([embedding, split_direct_conditions[i]], axis=1)
            preds.append(probability_net.posterior_probs(inference_net_input)[:, 1])
        m1_prob = np.concatenate(preds)
    else:
        m1_prob = np.array(
            probability_net.posterior_probs(summary_net(data["X"]))[:, 1],
            dtype=np.longdouble,
        )
    inference_time_stop = perf_counter()
    m0_prob = 1 - m1_prob
    selected_model = m1_prob > 0.5

    # Bayes Factors
    bayes_factor = m1_prob / m0_prob

    # Times
    training_time = np.repeat((training_time_stop - training_time_start), n_datasets)
    inference_time = np.repeat(
        ((inference_time_stop - inference_time_start) / n_datasets), n_datasets
    )

    # Final epoch mean loss
    final_epoch_loss = np.repeat(np.mean(losses[-1000:]), n_datasets)

    # Create DataFrame
    vals = np.c_[
        dataset,
        true_model,
        m0_prob,
        m1_prob,
        selected_model,
        bayes_factor,
        training_time,
        inference_time,
        final_epoch_loss,
    ]
    names = [
        "dataset",
        "true_model",
        "m0_prob",
        "m1_prob",
        "selected_model",
        "bayes_factor",
        "training_time",
        "inference_time",
        "final_epoch_loss",
    ]
    df = pd.DataFrame(vals, columns=names)
    df[["dataset", "true_model", "selected_model"]] = df[
        ["dataset", "true_model", "selected_model"]
    ].astype(int)

    return df


def log_with_inf_noise_addition(x, noise=0.000000001):
    """
    Adjusts the model probabilities leading to Inf values by a minimal amount of noise,
    recomputes the Bayes factors and then computes the log of the given array.
    """

    x_copy = x.copy()
    for i in range(x.shape[0]):
        if x.loc[i, "m0_prob"] == 0:
            print("Dataset with infinite BF: {}".format(i))
            x_copy.loc[i, "m0_prob"] = x_copy.loc[i, "m0_prob"] + noise
            x_copy.loc[i, "m1_prob"] = x_copy.loc[i, "m1_prob"] - noise
            x_copy.loc[i, "bayes_factor"] = (
                x_copy.loc[i, "m1_prob"] / x_copy.loc[i, "m0_prob"]
            )
    x_copy = np.log(x_copy["bayes_factor"])
    return x_copy


def computation_times(results_list):
    """
    Calculates the computation times of bridge sampling and a number of neural network variants.
    Assumes that the first result in results_list belongs to bridge sampling and the rest to the neural network(s).

    Parameters
    ----------
    results_list : list of pd.DataFrame objects
        Contains the results of bridge sampling and a number of neural network variants
    """

    # Calculate computation times
    results_time_list = []
    bridge_time = (
        results_list[0]["compile_time"]
        + (results_list[0]["stan_time"] + results_list[0]["bridge_time"]).cumsum()
    ) / 60
    results_time_list.append(bridge_time)

    for NN_result in results_list[1:]:
        results_time_list.append(
            (NN_result["training_time"] + NN_result["inference_time"].cumsum()) / 60
        )

    # Adjust index to represent datasets
    for i, result in enumerate(results_time_list):
        results_time_list[i].index += 1

    return results_time_list


# Levy flight application: Load and transform data


def load_simulated_rt_data(folder, indices_filename, datasets_filename):
    """Loads and transforms simulated reaction time data to cohere to BayesFlow data formatting.

    Parameters
    ----------
    folder : string
        Path to the folder containing the files
    indices_filename : string
    datasets_filename : string
    """

    indices = np.load(os.path.join(folder, indices_filename))
    datasets = np.load(os.path.join(folder, datasets_filename))

    # unpack indices
    indices = indices[:, 0, 0, 0] - 1

    # one-hot encode indices
    indices = to_categorical(indices, num_classes=4)

    return indices, datasets


def load_empirical_rt_data(load_dir):
    """
    Reads single subject datasets from a folder and transforms into list of 4D-arrays
    which allows for a variable number of observations between participants.
    Assumes data files have a three-column csv format (Condition, Response, Response Time).
    ----------

    Arguments:
    load_dir : str -- a string indicating the directory from which to load the data
    --------

    Returns:
    X: list of length (n_clusters), containing tf.Tensors of shape (1, 1, n_obs, 3)
        -- variable order now (Condition, Response Time, Response)
    """

    data_files = os.listdir(load_dir)
    X = []

    # Loop through data files and estimate
    for data_file in data_files:
        ### Read in and transform data
        data = pd.read_csv(os.path.join(load_dir, data_file), header=None, sep=" ")
        data = data[[0, 2, 1]].values  # reorder columns
        X_file = tf.convert_to_tensor(data, dtype=tf.float32)[np.newaxis][
            np.newaxis
        ]  # get 4D tensor
        X.append(X_file)

    return X


def mask_inputs(
    simulations_dict,
    missings_mean,
    missings_sd,
    missing_value=-1,
    missing_rts_equal_mean=True,
    insert_additional_missings=False,
):
    """Masks some training inputs so that training leads to a robust net that can handle missing data

    Parameters
    ----------
    simulations_dict : np.array
        simulated training data sets
    missings_mean : float
        empirical mean of missings per person
    missings_sd : float
        empirical sd of missings per person
    missing_rts_equal_mean : boolean
        indicates whether missings reaction time data should be imputed with the person mean
    insert_additional_missings : boolean
        indicates whether additional missings should be inserted, which results in disabling the faithfulness check

    Returns
    -------
    data_sets : np.array
        simulated training data sets with masked inputs
    """

    masked_dict = copy.deepcopy(simulations_dict)
    n_models = len(masked_dict['model_outputs'])
    n_data_sets_per_model = masked_dict['model_outputs'][0]['sim_data'].shape[0]
    n_persons = masked_dict['model_outputs'][0]['sim_data'].shape[1]
    n_trials = masked_dict['model_outputs'][0]['sim_data'].shape[2]

    # create truncated normal parameterization in accordance with scipy documentation
    a, b = (0 - missings_mean) / missings_sd, (n_trials - missings_mean) / missings_sd

    for m in range(n_models):
        for d in range(n_data_sets_per_model):
            # draw number of masked values per person from truncated normal distribution
            masks_per_person = (
                truncnorm.rvs(a, b, loc=missings_mean, scale=missings_sd, size=n_persons)
                .round()
                .astype(int)
            )
            # assign the specific trials to be masked within a person
            mask_positions = [
                np.random.randint(0, n_trials, size=(n_persons, j))
                for j in masks_per_person
            ][0]
            for j in range(n_persons):
                masked_dict['model_outputs'][m]['sim_data'][d, j, :, 1:3][mask_positions[j]] = missing_value
                if missing_rts_equal_mean:
                    masked_dict['model_outputs'][m]['sim_data'][d, j, :, 1][mask_positions[j]] = np.mean(
                        masked_dict['model_outputs'][m]['sim_data'][d, j, :, 1]
                    )

    # assert that the average amount of masks per person matches the location of the truncnormal dist
    if insert_additional_missings == False:
        for m in range(n_models):
            deviation = abs(
                (masked_dict['model_outputs'][m]['sim_data'][:, :, :, :] == missing_value).sum()
                / (n_data_sets_per_model * n_persons * (2 - missing_rts_equal_mean))
                - missings_mean
            )
            assert (
                deviation < 3
            ), f"Average amount of masks per person deviates by {deviation} from missings_mean!"

    return masked_dict


def join_and_fill_missings(
    color_data, lexical_data, n_trials, missings_value=-1, missing_rts_equal_mean=True
):
    """Joins data from color discrimination and lexical decision task per person and fills missings

    Parameters
    ----------
    color_data : tf.Tensor
    lexical_data : tf.Tensor
    n_trials : int
    missings_value : float
        specifies the value that codes missings
    missing_rts_equal_mean : boolean
        specifies whether missing rt should be coded with missings_value or imputed with the participant mean

    Returns
    -------
    empirical_data : list of tf.Tensors
    """

    n_clusters = len(color_data)
    n_trials_per_cond = int(n_trials / 2)
    empirical_data = []

    for j in range(n_clusters):
        # Join data
        joint_data = tf.concat([color_data[j], lexical_data[j]], axis=2).numpy()
        # Extract information about trial
        n_trials_obs = joint_data.shape[2]
        n_missings = n_trials - n_trials_obs
        n_condition_1 = int(joint_data[0, 0, :, 0].sum())
        mean_rt = np.mean(joint_data[0, 0, :, 1])
        # replace all missings with missings_value
        npad = ((0, 0), (0, 0), (0, n_missings), (0, 0))
        joint_data = np.pad(
            joint_data, npad, "constant", constant_values=missings_value
        )
        # replace missing condition indices
        cond_indices = np.array(
            [0] * (n_trials_per_cond - (n_trials_obs - n_condition_1))
            + [1] * (n_trials_per_cond - n_condition_1)
        )
        np.random.shuffle(cond_indices)
        joint_data[0, 0, -(n_missings):, 0] = cond_indices
        # replace missing rts with mean rt
        if missing_rts_equal_mean:
            joint_data[0, 0, :, 1] = np.select(
                [joint_data[0, 0, :, 1] == missings_value],
                [mean_rt],
                joint_data[0, 0, :, 1],
            )
        # Append
        empirical_data.append(joint_data)

    # Transform to np.array
    empirical_data = np.reshape(
        np.asarray(empirical_data), (1, n_clusters, n_trials, 3)
    )

    # Assert that the number of coded missings equals the real number of missings
    deviation = abs(
        (
            (empirical_data == missings_value).sum()
            / (n_clusters * (2 - missing_rts_equal_mean))
        )
        - 28.5
    )
    assert deviation < 1, "number of filled and existing missings does not match"

    return empirical_data


# Lévy flight application: Robustness against additional noise


def mean_predictions_noisy_data(
    empirical_data, probability_net, summary_net, missings_mean, n_runs
):
    """Get mean predictions for repeatedly applying the network to nested data with a proportion randomly masked as missing.
        Variation of missing value between persons is held minimal via a low missings_sd parameter inside the runs loop.

    Parameters
    ----------
    empirical_data : np.array
        4-dimensional array with shape (n_datasets, n_clusters, n_obs, n_variables).
    probability_net : ModelProbabilityNetwork
        Trained model probability network.
    summary_net : HierarchicalInvariantNetwork
        Trained summary network.
    missings_mean : float
        Mean number of missing trials that should be added to the data.
        (Can overlap with existing missings, resulting in less additional missing trials than stated)
    n_runs : int
        Number of runs to average over.


    Returns
    -------
    mean_noise_proportion : float
        Mean proportion of missing trials in the data.
    mean_probs : np.array
        Mean predictions output by the model probability network.
    mean_probs_sds : np.array
        Standard deviations of the mean predictions over the runs.
    mean_vars : np.array
        Mean variability output by the model probability network.
    """

    noise_proportion = []
    probs = []

    n_clusters = empirical_data.shape[1]
    n_obs = empirical_data.shape[2]

    for r in range(n_runs):
        noisy_data = mask_inputs(
            empirical_data,
            missings_mean=missings_mean,
            missings_sd=0.0001,
            missing_rts_equal_mean=True,
            insert_additional_missings=True,
        )
        noise_proportion_run = (noisy_data == -1).sum() / (n_clusters * n_obs)
        noise_proportion.append(noise_proportion_run)
        preds = probability_net.predict(summary_net(noisy_data))
        probs.append(preds["m_probs"])

    mean_noise_proportion = np.mean(noise_proportion)
    mean_probs = np.mean(probs, axis=0)
    mean_probs_sds = np.std(probs, axis=0)

    return mean_noise_proportion, mean_probs, mean_probs_sds


def inspect_robustness_noise(
    added_noise_percentages, empirical_data, probability_net, summary_net, n_runs
):
    """Utility function to inspect the robustness of the network to artificially added noise.

    Parameters
    ----------
    added_noise_percentages : np.array
        Array of noise percentage steps.
    empirical_data : np.array
        4-dimensional array with shape (n_datasets, n_clusters, n_obs, n_variables).
    probability_net : ModelProbabilityNetwork
        Trained model probability network.
    summary_net : HierarchicalInvariantNetwork
        Trained summary network.
    n_runs : int
        Number of runs to average over.

    Returns
    -------
    mean_noise_proportion_list : list
        Mean noise proportions in the data sets for each noise step.
        Can deviate from added_noise_percentages as added noise can overlap with existing missings.
    mean_probs : np.array
        Mean predictions output by the model probability network for each noise step.
    mean_sds : np.array
        Standard deviations of the mean predictions over the runs for each noise step.
    """

    mean_noise_proportion_list = []
    means_probs_list = []
    means_probs_sds_list = []

    for noise in added_noise_percentages:
        missings_mean = 900 * noise
        mean_noise_proportion, mean_probs, mean_probs_sds = mean_predictions_noisy_data(
            empirical_data,
            probability_net,
            summary_net,
            missings_mean=missings_mean,
            n_runs=n_runs,
        )
        mean_noise_proportion_list.append(mean_noise_proportion)
        means_probs_list.append(mean_probs)
        means_probs_sds_list.append(mean_probs_sds)

    mean_probs = np.squeeze(means_probs_list)
    mean_sds = np.squeeze(means_probs_sds_list)

    return mean_noise_proportion_list, mean_probs, mean_sds


# Lévy flight application: Robustness against bootstrapping


def inspect_robustness_bootstrap(
    empirical_data, probability_net, summary_net, level, n_bootstrap=100
):
    """Utility function to inspect the robustness of the network to bootstrapping.

    Parameters
    ----------
    empirical_data : np.array
        4-dimensional array with shape (n_datasets, n_clusters, n_obs, n_variables).
    probability_net : ModelProbabilityNetwork
        Trained model probability network.
    summary_net : HierarchicalInvariantNetwork
        Trained summary network.
    level : string
        Indicating the level to bootstrap; either 'participants' or 'trials'.
    n_bootstrap : int
        Number of bootstrap repetitions.

    Returns
    -------
    bootstrap_probs : np.array
        Predictions on the bootstrapped data sets output by the model probability network.
    """

    if level == "participants":
        n = empirical_data.shape[1]
    elif level == "trials":
        n = empirical_data.shape[2]

    bootstrapped_probs = []

    for b in range(n_bootstrap):
        b_idx = np.random.choice(np.arange(n), size=n, replace=True)
        if level == "participants":
            bootstrapped_data = empirical_data[:, b_idx, :, :]
        elif level == "trials":
            bootstrapped_data = empirical_data[:, :, b_idx, :]
        probs = probability_net.predict(summary_net(bootstrapped_data))["m_probs"]
        bootstrapped_probs.append(probs)

    bootstrapped_probs = np.asarray(bootstrapped_probs)[:, 0, :]

    return bootstrapped_probs


# Lévy flight application: Robustness against leaving single participants out (LOPO)


def inspect_robustness_lopo(
    empirical_data, probability_net, summary_net, print_probs=False
):
    """Utility function to inspect the robustness of the network to leaving single participants out (LOPO).

    Parameters
    ----------
    empirical_data : np.array
        4-dimensional array with shape (n_datasets, n_clusters, n_obs, n_variables).
    probability_net : ModelProbabilityNetwork
        Trained model probability network.
    summary_net : HierarchicalInvariantNetwork
        Trained summary network.
    print : boolean
        Optional: Print predictions per participant to find influential participants.

    Returns
    -------
    lopo_probs : np.array
        Predictions on the LOPO data sets output by the model probability network.
    """

    n_participants = empirical_data.shape[1]

    lopo_probs = []

    for b in range(n_participants):
        cropped_data = np.delete(empirical_data, b, axis=1)
        probs = probability_net.predict(summary_net(cropped_data))["m_probs"]
        lopo_probs.append(probs)
        if print_probs:
            print_probs(f"Participant = {b+1} / Predictions  = {probs}")

    lopo_probs = np.asarray(lopo_probs)[:, 0, :]

    return lopo_probs
