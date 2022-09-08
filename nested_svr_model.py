import pandas as pd
import numpy as np
import os
import time
import datetime
import warnings
from tqdm import tqdm
import scipy.stats as stats
from sklearn.svm import LinearSVR, LinearSVC, SVR, SVC
from joblib import Parallel, delayed
from helper_functions import run_model, run_model_inner_kfold
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")


def nested_svr_basic(X, y, leftout, p_threshold, C_range, inner_folds, covariate, control_method, random_seed, if_permutation, permutation_seed):

    """Sub-function that will be used in nested_svr_model for parallel computing
      - X: 3d functional connectivity matrix (n*n*m, n represents the number of nodes, m is number of subjects)
      - y: SPE score of all subjects
      - p_threshold: the p threshold which will be applied to select features
      - C_range: a list of alternative values for hyperparameter C tuning
      - inner_folds: int, such as 3, 5, 10; k-fold in the inner loop
      - covariate: list, values of a covariate, valid only if control_method != none
      - control_method: None, partial, or delete. For details see the article
      - random_seed: for reproducibility, random seed need to be fixed
      - if_permutation: permutation or not
      - permutation_seed: for reproducibility, random seed of permutation need to be fixed in each permutation
    """

    # get the training set
    X_train = np.delete(X, leftout, axis=2)
    y_train = np.delete(y, leftout)

    # permutation or not
    if if_permutation:
        y_train = np.array(pd.DataFrame(y_train).sample(frac=1, replace=False, random_state=permutation_seed)).reshape(-1,)

    # kfold cv for the training set to obtain the best hyperparameter
    temp_results = pd.DataFrame()
    i = 0
    for C in C_range:
        # create a model
        model = LinearSVR(C=C, random_state=random_seed)

        # run the model
        y_oof, prediction_oof = run_model_inner_kfold(model, X_train, y_train, inner_folds, p_threshold, random_seed, if_permutation)

        # evaluate the performance
        print(C)
        [r, p] = stats.pearsonr(prediction_oof, y_oof)
        temp_results.loc[i, 'threshold'] = p_threshold
        temp_results.loc[i, 'C'] = C
        temp_results.loc[i, 'r'] = r
        i += 1

    # get the best C with best performance
    temp_results = temp_results.sort_values('r', ascending=False).reset_index(drop=True)
    best_C = temp_results.loc[0, 'C']

    # use the whole training set to predict on test set (i.e., y for the subject been left out)
    best_model = LinearSVR(C=best_C, random_state=random_seed)
    run_model(best_model, X, y, leftout, p_threshold, covariate, control_method, if_permutation, permutation_seed)


def nested_svr_model(X, y, covariate, p_threshold, C_range, inner_folds, control_method, control_threshold, random_seed=0, parallel_quantity=-1, if_permutation=False, permutation_seed=0):

    """Main function of the nested-LOOCV prediction
      - X: 3d functional connectivity matrix (n*n*m, n represents the number of nodes, m is the number of subjects)
      - y: SPE score of all subjects
      - covariate: list, values of a covariate, valid only if control_method != none
      - p_threshold: the p threshold which will be applied to select features
      - C_range: a list of alternative values for hyperparameter C tuning
      - inner_folds: int, such as 3, 5, 10; k-fold in the inner loop
      - control_method: None, partial, or delete. For details see the article
      - control_threshold: When control_method = delete, apply this threshold to delete relevant features
      - random_seed: for reproducibility, random seed need to be fixed
      - parallel_quantity: how many threads will be employed when parallel computing
      - if_permutation: permutation or not
      - permutation_seed: for reproducibility, random seed of permutation need to be fixed in each permutation
    """

    if if_permutation:
        print(f'Start running with permutation seed = {permutation_seed}:  ', datetime.datetime.now().strftime('%H:%M:%S'))
    no_sub = X.shape[2]
    no_node = X.shape[0]
    assert no_sub == len(y)

    # when need to remove those features associated with covariates to verify model robustness
    if control_method == 'delete':

        X_2D = X.reshape((no_node * no_node, no_sub)).T

        # delete the edges that are correlated with y
        ps = []
        for i in range(X_2D.shape[1]):
            if len(set(covariate)) == 2:
                # if the covariate is dichotomous (e.g., gender), use the t-test
                if i == 0:
                    print("t test is performing")
                edge = X_2D[:, i]
                edge_group1 = edge[covariate == list(set(covariate))[0]]
                edge_group2 = edge[covariate == list(set(covariate))[1]]
                t, p = stats.ttest_ind(edge_group1, edge_group2)
                ps.append(p)
            else:
                (r, p) = stats.pearsonr(X_2D[:, i], covariate)
                ps.append(p)

        # remove the features accordingly (let their values equal zero)
        ps_mask = np.array(ps) < control_threshold
        print(f"Delete {np.sum(ps_mask)} edges under controlled covariate")
        X_2D[:, ps_mask] = 0
        X = X_2D.T.reshape(no_node, no_node, no_sub)

    # use parallel processing to run the model
    Parallel(n_jobs=parallel_quantity, backend="loky")(delayed(nested_svr_basic)(X, y, leftout, p_threshold, C_range, inner_folds, covariate, control_method, random_seed, if_permutation, permutation_seed)
                                                       for leftout in range(no_sub))

    # get the prediction results
    y_predictions = np.zeros((no_sub, ))
    r2_all = []
    feature_nums = []
    for k in range(no_sub):
        y_predictions[k] = np.load(f'output/cache/svr/{k}.npz')['a']
        r2_all.append(np.load(f'output/cache/svr/{k}.npz')['b'])
        feature_nums.append(np.load(f'output/cache/svr/{k}.npz')['c'])
        os.remove(f'output/cache/svr/{k}.npz')

    # save the predictions
    df_results = pd.DataFrame()
    df_results['prediction'] = y_predictions
    df_results['y'] = y
    df_results.to_csv(f"output/nest_svr_predictions_{int(time.time())}.csv", index=False)

    # evaluate model performance
    [r, p] = stats.pearsonr(y_predictions, y)
    print(f'model performance: r = {r}, p = {p}, MAE={mean_absolute_error(y, y_predictions)}')
