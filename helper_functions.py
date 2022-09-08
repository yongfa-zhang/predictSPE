import numpy as np
import pandas as pd
import scipy.stats as stats
import os
from sklearn import preprocessing
from sklearn.model_selection import KFold, train_test_split


def get_atlas_node_template(node_cnt):
    """Generate a feature template for later analyses.  node_cnt: number of nodes defined in the atlas."""
    feature_id = list(range(node_cnt * node_cnt))
    node_x = []
    node_y = []

    for i in range(node_cnt):
        node_x += list(np.ones(node_cnt) * i)
        node_y += list(range(node_cnt))

    template = pd.DataFrame({"feature_id": feature_id, "node_x": node_x, "node_y": node_y}).astype(np.int64)
    return template


def haufe_transformation(X, coef):
    """Transform the raw coefficients into interpretable activation patterns according to Haufe's(2014) definition."""
    y_hat_calculated = X.dot(coef.T)
    cov_y_hat = np.cov(y_hat_calculated, rowvar=False)
    cov_x = np.cov(X, rowvar=False)

    return cov_x.dot(coef.T) / cov_y_hat


def partial_corr_calculator(x, y, covar):
    """calculate the partial correlation between x and y, while controls [covar]"""
    assert len(x) == len(y) == len(covar)
    n = len(x)
    freedom_degree = n - 3
    (r_xy, p1) = stats.pearsonr(x, y)
    (r_xc, p2) = stats.pearsonr(x, covar)
    (r_yc, p3) = stats.pearsonr(y, covar)

    r = (r_xy-r_xc*r_yc) / ((1-r_xc**2)**0.5 * (1-r_yc**2)**0.5)

    t = r / ((1 - r**2) / freedom_degree) ** 0.5
    p = stats.t.sf(abs(t), freedom_degree) * 2

    return (r, p)


def run_model(model, X, y, leftout, threshold, covariate, control_method, if_permutation, permutation_seed):

    """Given a defined model and data, this function can generate the prediction result of the left out subject
          - model: a defined model (e.g., svr)
          - X: 3d functional connectivity matrix (n*n*m, n represents the number of node, m is number of subjects)
          - y: SPE score of all subjects
          - leftout: the index of the leftout subject in the LOOCV
          - threshold: the p threshold which will be applied to select features
          - covariate: list, values of a covariate, valid only if control_method != none
          - control_method: None, partial, or delete. For details see the article
          - if_permutation: permutation or not
          - permutation_seed: for reproducibility, random seed of permutation need to be fixed in each permutation
    """

    # get the training set and test set
    no_sub = X.shape[2]
    no_node = X.shape[0]

    X_test = X[:, :, leftout].copy()
    X_test_2D = X_test.reshape((no_node * no_node, 1)).T
    y_test = y[leftout]
    X_train = np.delete(X, leftout, axis=2)
    X_train_2D = X_train.reshape((no_node * no_node, no_sub - 1)).T
    y_train = np.delete(y, leftout)

    # permutation or not
    if if_permutation:
        y_train = np.array(pd.DataFrame(y_train).sample(frac=1, replace=False, random_state=permutation_seed)).reshape(-1,)

    # calculate the Pearson correlation coef. for each feature
    rs = []
    ps = []
    for i in range(X_train_2D.shape[1]):
        if control_method == 'partial':
            # when using partial correlation to test the model robustness on covariates
            covariate_train = np.delete(np.array(covariate), leftout)
            [r, p] = partial_corr_calculator(X_train_2D[:, i], y_train, covariate_train)
        else:
            (r, p) = stats.pearsonr(X_train_2D[:, i], y_train)
        rs.append(r)
        ps.append(p)

    # select the most relevant features
    mask = np.array(ps) < threshold
    # notice, when permutation test, there may not be enough relevant features, so we can pick the top 50 features for prediction
    if if_permutation and sum(mask) < 50:
        mask[:] = False
        mask[list(np.array(ps).argsort()[:50])] = True

    X_train_2D = X_train_2D[:, mask]
    X_test_2D = X_test_2D[:, mask]

    # deduplicate columns for the feature matrices
    X_train_2D, u_index = np.unique(X_train_2D, axis=1, return_index=True)
    X_test_2D = X_test_2D[:, u_index]

    # normalize the features
    normalize = preprocessing.MinMaxScaler()
    X_train_2D = normalize.fit_transform(X_train_2D)
    X_test_2D = normalize.transform(X_test_2D)

    # fit the model
    model.fit(X_train_2D, y_train)
    y_prediction = model.predict(X_test_2D)[0]
    r2_training_set = model.score(X_train_2D, y_train)

    # transform the raw coef. to interpretable activation patterns, for details see haufe (2014)
    activation_patterns = haufe_transformation(X_train_2D, model.coef_)

    # save data for subsequent interpretation
    np.savez(f'output/svr_features/features_{leftout}.npz', mask=mask, index=u_index, rs=rs, coef=model.coef_, act=activation_patterns)

    # save the prediction (for parallel processing)
    if not os.path.exists('output/cache/svr/'):
        os.mkdir('output/cache/svr/')
    np.savez(f'output/cache/svr/{leftout}.npz', a=y_prediction, b=r2_training_set, c=X_train_2D.shape[1])


def run_model_inner_kfold(model, X, y, fold_k, threshold, random_seed, if_permutation):

    """Given a defined model and data, this function can use k-fold CV to generate prediction results
          - model: a defined model (e.g., svr)
          - X: 3d functional connectivity matrix (n*n*m, n represents the number of node, m is number of subjects)
          - y: SPE score of all subjects
          - fold_K: such as 3, 5, 10
          - threshold: the p threshold which will be applied to select features
          - random_seed: random seed for shuffling data during train_test_split
          - if_permutation: permutation or not
    """

    # transfer the original 3d-feature matrix to 2d
    no_sub = X.shape[2]
    no_node = X.shape[0]
    X_2D = X.reshape((no_node * no_node, no_sub)).T

    # k-fold cross validation
    X_train_2D, X_test_2D, y_train, y_test = train_test_split(X_2D, y, test_size=1/fold_k, random_state=random_seed)

    # deduplicate columns for the feature matrices
    X_train_2D, u_index = np.unique(X_train_2D, axis=1, return_index=True)
    X_test_2D = X_test_2D[:, u_index]

    # normalize the features
    normalize = preprocessing.MinMaxScaler()
    X_train_2D = normalize.fit_transform(X_train_2D)
    X_test_2D = normalize.transform(X_test_2D)

    # calculate the Pearson correlation coef. for each feature, then select the most relevant features
    rs = []
    ps = []
    for i in range(X_train_2D.shape[1]):
        (r, p) = stats.pearsonr(X_train_2D[:, i], y_train)
        rs.append(r)
        ps.append(p)

    mask = np.array(ps) < threshold

    # notice, when permutation test, there may not be enough relevant features, so we can pick the top 50 features for prediction
    if if_permutation and sum(mask) < 50:
        mask[:] = False
        mask[list(np.array(ps).argsort()[:50])] = True

    X_train_2D = X_train_2D[:, mask]
    X_test_2D = X_test_2D[:, mask]

    # fit model
    model.fit(X_train_2D, y_train)
    y_prediction = model.predict(X_test_2D)

    return y_test, y_prediction

