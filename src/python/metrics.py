import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


def performance_metrics(results_list, names, metrics, model_prior):
    """
    Calculates various metrics to compare approximation performance.
    Be careful about model_prior: can diverge from true proportion!

    Parameters
    ----------
    results_list : list of pd.DataFrame objects
        Contains the results of bridge sampling and a number of neural network variants
    names : list
        Method names

    Returns
    -------
    df : pd.DataFrame
        Table with calculated performance metrics
    """

    accuracy = []
    roc_auc = []
    mae = []
    rmse = []
    log_score = []
    bias = []

    for d in results_list:
        accuracy_temp = (d["true_model"] == d["selected_model"]).mean()
        accuracy.append(accuracy_temp)

        roc_auc_temp = roc_auc_score(d["true_model"], d["m1_prob"])
        roc_auc.append(roc_auc_temp)

        mae_temp = np.mean(abs(d["true_model"] - d["m1_prob"]))
        mae.append(mae_temp)

        rmse_temp = np.sqrt(((d["true_model"] - d["m1_prob"]) ** 2).mean())
        rmse.append(rmse_temp)

        log_score_temp = log_loss(d["true_model"], d["m1_prob"])
        log_score.append(log_score_temp)

        bias_temp = model_prior - d["m1_prob"].mean()
        bias.append(bias_temp)

    df = pd.DataFrame(
        [accuracy, roc_auc, mae, rmse, log_score, bias], index=metrics, columns=names
    ).transpose()

    return df


def bootstrapped_metrics(results_list, n_bootstrap, names, metrics, model_prior):
    """
    Calculates bootstrapped performance metrics.
    Be careful about model_prior: can diverge from true proportion!

    Parameters
    ----------
    results_list : list of pd.DataFrame objects
        Contains the results of bridge sampling and a number of neural network variants
    n_bootstrap : integer
        Number of bootstrap steps
    names : list
        Method names
    """

    n_test = results_list[0].shape[0]  # number of test data sets

    perf_metrics_bootstrapped = []

    for b in range(n_bootstrap):
        b_idx = np.random.choice(np.arange(n_test), size=n_test, replace=True)
        results_list_bootstrapped = [x.iloc[b_idx, :] for x in results_list]
        perf_metrics = performance_metrics(
            results_list_bootstrapped, names, metrics, model_prior
        )
        perf_metrics_bootstrapped.append(perf_metrics)

    bootstrapped_mean = pd.DataFrame(
        np.mean(perf_metrics_bootstrapped, axis=0), index=names, columns=metrics
    )
    bootstrapped_ses = pd.DataFrame(
        np.std(perf_metrics_bootstrapped, axis=0), index=names, columns=metrics
    )

    return bootstrapped_mean, bootstrapped_ses
