import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error as MAE,
    mean_squared_error as RMSE,
    r2_score as R2,
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
from sklearn.model_selection import GridSearchCV, ParameterGrid, ShuffleSplit
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import time
from sklearn.ensemble import RandomForestRegressor
from pickle import dump
from sklearn.dummy import DummyRegressor
import textwrap
from matplotlib.ticker import MaxNLocator
from sklearn.base import clone
from shared import (
    PickleableKerasRegressor,
    HistoryKerasRegressor,
    model_fun,
    reload_model,
    compute_metrics
)
from extra import ding

resources = "resources/"
memory = ".memory"
tv_split_path = os.path.join(resources, "tv_split.csv.gz")
test_split_path = os.path.join(resources, "test_split.csv.gz")

# Load data and return feature matrix and target vector
def load_data(path):
    print(f"Loading {path}...")
    df = pd.read_csv(path)
    y = df[["GPP"]]
    X = df.drop(["GPP", "site", "date"], axis=1)
    print(X.shape)
    print(y.shape)
    return X, y


# Train a griven model, compute the metrics on the train/validation and test sets,
# and persist the model to disk for later use.
def train_and_test(model_name, model, X_tv, y_tv, X_test, y_test, fit_params={}):
    # Fit the model
    start = time.time()
    model.fit(X_tv, y_tv, **fit_params)
    end = time.time()
    train_time = round(end - start, 2)
    # Print mean fit time
    print(f"Total training time: {train_time}s")
    try:
        cv_results = pd.DataFrame(model.cv_results_)
        print(f"Mean fit time: {cv_results['mean_fit_time'].mean().round(2)}s")
    except:
        pass
    # Compute the metrics
    compute_metrics(model, X_tv, y_tv, "train")
    compute_metrics(model, X_test, y_test, "test")
    if model_name:
        # Persist model to disk
        with open(os.path.join(resources, f"{model_name}.pkl"), "wb") as f:
            dump(model, f, protocol=5)
    return train_time


# Format a string using a text wrapper
def formatter(lbl, pos=0):
    return textwrap.fill(lbl, 16)


# Display a hyperparameter graph
def display_graph(
    model,
    hyperparam_name,  # Hyperparameter name
    xlog=False,  # Whether X axis should be logarithmic
    xint=False,  # Whether the X axis should be integers
):
    plt.figure(figsize=(8, 4))
    cv_results = pd.DataFrame(model.cv_results_)
    is_obj = False
    # We group the results by `hyperparam_name`
    # The reason for the try/except is to handle cases where the hyperparameter is not numeric (ex: a scaler)
    try:
        cv_results["group"] = cv_results["param_regressor__" + hyperparam_name]
        grouped = cv_results.groupby("group")
        params = grouped.indices.keys()
    except:
        is_obj = True
        cv_results["group"] = cv_results["param_regressor__" + hyperparam_name].astype(
            str
        )
        grouped = cv_results.groupby("group")
        params = [formatter(p) for p in grouped.indices.keys()]

    # get the index of the best samples (max of mean test score in each group)
    idx_best_samples = grouped["mean_test_score"].idxmax()
    # get the list of the best samples
    best_samples = idx_best_samples.apply(lambda idx: cv_results.loc[idx])

    # Plot mean scores
    plt.plot(params, best_samples["mean_train_score"], label="train")
    plt.plot(params, best_samples["mean_test_score"], label="validation")

    # Add marker for best score
    best_param = best_samples["mean_test_score"].idxmax()
    best_run = best_samples.loc[best_param]
    best_score = best_run["mean_test_score"]
    # We format the string when needed
    if is_obj:
        best_param_formated = formatter(best_param)
    else:
        best_param_formated = best_param
    plt.scatter(best_param_formated, best_score, marker="x", c="red", zorder=10)

    if xlog:
        plt.xscale("log")

    # Quantify variance with ±std curves
    n_sigmas = 2
    plt.fill_between(
        params,
        best_samples["mean_train_score"] - n_sigmas * best_samples["std_train_score"],
        best_samples["mean_train_score"] + n_sigmas * best_samples["std_train_score"],
        alpha=0.2,
    )
    plt.fill_between(
        params,
        best_samples["mean_test_score"] - n_sigmas * best_samples["std_test_score"],
        best_samples["mean_test_score"] + n_sigmas * best_samples["std_test_score"],
        alpha=0.2,
    )

    # Some mre formating
    try:
        val = best_param.round(2)
    except:
        val = best_param

    if xint:
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.title(f"Best {hyperparam_name}: {val} (score: {best_score.round(2)})")
    plt.ylabel("Score")
    plt.xlabel(hyperparam_name)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
# Helper function to create grid dicts
def make_grid_impl(scalers, grid_base_prefixed):
    for scaler in scalers:
        yield {
            **grid_base_prefixed,
            "regressor__scaler": [scaler],
            "transformer": [clone(scaler) if scaler else None],
        }


# Creates a grid given a list of scalers and a grid dict
# The scaler is applied to both the TransformedTargetRegressor and the data pre-processing scaler
def make_grid(scalers, grid_regressor):
    grid_regressor_prefixed = {"regressor__" + k: v for k, v in grid_regressor.items()}
    return list(make_grid_impl(scalers, grid_regressor_prefixed))


# Build, train and test a model
def eval_model(
    name,  # Model name
    regressor,  # Regressor to use in the pipeline
    grid,  # Parameter grid
    X_tv, y_tv, X_test, y_test,
    features=None,  # Subset of features to use. If None, all features are used
    pca=False,  # If True, a PCA is inserted before the regressor
    n_splits=3,  # Number of splits. A ShuffleSplit is used with test_size of 0.3
    train_size=None,  # Size of the test splits (number of samples per split). If None, complement of the test_size
    scoring="neg_mean_absolute_error",
    n_jobs=4,
    verbose=1,
):
    if train_size:
        # We convert the number of samples into a value between 0 and 1
        train_size = train_size / len(X_tv)

    # Transformer to select only a subset of features
    if features:
        ct = ColumnTransformer(
            [("selector", "passthrough", features)], remainder="drop"
        )
    else:
        ct = None

    # PCA
    if pca:
        pca_tr = PCA()
        # We use memory to speed up the preprocessing
        mem = memory
    else:
        pca_tr = None
        mem = None

    # Pipeline including the regressor provided in as a parameter
    reg = Pipeline(
        [("selector", ct), ("scaler", None), ("pca", pca_tr), ("reg", regressor)],
        memory=mem,
    )

    # Wrap the pipeline in a TransformedTargetRegressor to scale the target variable
    tr_regressor = TransformedTargetRegressor(regressor=reg, transformer=None)
    #print(tr_regressor)

    # Wrap into a GridSearchCV
    model = GridSearchCV(
        tr_regressor,
        grid,
        cv=ShuffleSplit(n_splits, test_size=0.3, train_size=train_size),
        return_train_score=True,
        scoring=scoring,
        #refit=scoring[0],
        verbose=verbose,
        n_jobs=n_jobs,
    )

    # Train and test the model
    fitting_time = train_and_test(name, model, X_tv, y_tv, X_test, y_test)
    time_obj = time.gmtime(fitting_time)
    
    # Notify
    if name:
        ding(f"Model {name} completed in {time.strftime('%H:%M:%S', time_obj)}")

    return model
