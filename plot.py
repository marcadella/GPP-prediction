import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from shared import reload_model, compute_metrics

model_list = [
    "baseline",
    "ridge_2p_rs",
    "knn_2p_pca",
    "ridge_full",
    "rf_full",
    "svr_full",
    "dnn_full",
]

# Add "True GPP" to the list
def models_and_true(models):
    if "True GPP" not in models:
        return ["True GPP"] + models
    else:
        return models

# Plot the prediction histogram and scatter plot (vs true value)
def pred_plot(predictions, model, sample_frac=1):
    p = predictions.sample(frac=sample_frac)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(data=p[["True GPP", model]], bins=20, kde=True, ax=axes[0])
    line_x = [predictions["True GPP"].min(), predictions["True GPP"].max()]
    axes[1].plot(line_x, line_x, c="gray")
    sns.scatterplot(x=p["True GPP"], y=p[model], ax=axes[1], alpha=0.7)
    axes[0].set_xlabel("GPP")
    axes[0].set_title("True and predicted densities")
    axes[1].set_xlabel("True GPP")
    axes[1].set_ylabel("Predicted GPP")
    axes[1].set_title("Comparision of predictions against ground truth")
    plt.suptitle(f"{model} analysis")
    plt.tight_layout()
    plt.show()
    

# Plot the prediction error histogram and scatter plot (vs true value)
def err_plot(predictions, errors, model):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(data=errors[model], bins=20, kde=True, ax=axes[0])
    line_x = np.array([predictions["True GPP"].min(), predictions["True GPP"].max()])
    axes[1].plot(line_x, 0 * line_x, c="gray")
    sns.scatterplot(x=predictions["True GPP"], y=errors[model], ax=axes[1], alpha=0.7)
    axes[0].set_xlabel("Prediction error")
    axes[0].set_title("Error densities")
    axes[1].set_xlabel("True GPP")
    axes[1].set_ylabel("Prediction error")
    axes[1].set_title("Errors against their target")
    plt.suptitle(f"{model} error analysis")
    plt.tight_layout()
    plt.show()

    
# Plot the monthly and per site prediction error densities
def error_per_month_and_site(errors, model):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    sns.barplot(
        x=errors.index.get_level_values(1).month, y=model, data=errors, ax=axes[0]
    )
    sorted_index = errors[model].groupby("site").mean().sort_values().index.values
    sns.barplot(
        x=errors.index.get_level_values(0),
        y=model,
        data=errors,
        order=sorted_index,
        ax=axes[1],
    )
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Average prediction error")
    axes[1].set_xlabel("Site")
    axes[1].set_ylabel("Average prediction error")
    plt.title(f"{model} errors per month and site")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

# Plot the monthly and yearly prediction error densities
def error_per_month_and_year(errors, model):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    sns.violinplot(x=errors.index.month, y=model, data=errors, ax=axes[0])
    sns.violinplot(
        x=errors.index.year,
        y=model,
        data=errors,
        ax=axes[1],
    )
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Prediction error")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Prediction error")
    plt.suptitle(f"Monthly and yearly error repartitions (model: {model})")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

# Plot the predicted densities for all provided models
def plot_densities(predictions, models=model_list[1:], cover_type=None):
    plt.figure(figsize=(10, 6))
    models = models_and_true(models)
    for c in models:
        sns.kdeplot(predictions[c], label=c)
    plt.xlabel("GPP")
    plt.legend()
    if cover_type:
        plt.title(f"Comparision of the model densities for {cover_type}")
    else:
        plt.title("Comparision of the model densities")
    plt.show()

# Plot violin plot of the error for all provided models
def plot_errors(errors, models=model_list):
    plt.figure(figsize=(7, 5))
    sns.violinplot(data=errors[models])
    plt.title("Comparision of model errors")
    plt.ylabel("Prediction error")
    plt.tight_layout()
    plt.show()
    
# Plot predicted densities and violin plot for all provided models
def plot_desity_and_errors(predictions, errors, models=model_list[1:], cover_type=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for m in models_and_true(models):
        sns.kdeplot(predictions[m], label=m, ax=axes[0])
    axes[0].set_xlabel("GPP")
    axes[0].legend()
    axes[0].set_title(f"Comparision of the model densities")
    sns.violinplot(data=errors[["baseline"] + models], ax=axes[1])
    axes[1].set_title("Comparision of model errors")
    axes[1].set_ylabel("Prediction error")
    for tick in axes[1].get_xticklabels():
        tick.set_rotation(25)
    if cover_type:
        plt.suptitle(f"Predicted densities and errors for {cover_type}")
    else:
        plt.suptitle("Predicted densities and errors")
    plt.tight_layout()
    plt.show()
    
# Plot a bar chart summeraizing the metrics for all the models
def plot_metrics(results, labels, sort_by=None):
    if not sort_by:
        sort_by = labels[-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    mae = ["MAE_" + label for label in labels]
    rmse = ["RMSE_" + label for label in labels]
    results[mae].sort_values(f"MAE_{sort_by}").plot.bar(ax=axes[0])
    results[rmse].sort_values(f"RMSE_{sort_by}").plot.bar(ax=axes[1])
    axes[0].set_xlabel("Model")
    axes[1].set_xlabel("Model")
    plt.suptitle("Metrics for all the models")
    plt.tight_layout()
    plt.show()

# Plot the predicted and true value, and error timelines
def plot_timeline(predictions, errors, model, site_type):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    axes[0].plot(
        predictions[["True GPP", model]], ".", label=["True GPP", "Prediction"]
    )
    axes[1].plot(errors[model], ".")
    axes[0].legend()
    axes[0].set_ylabel("GPP")
    axes[1].set_title("Timeline errors")
    axes[1].set_ylabel("Error")
    axes[1].set_title("Timeline predictions")
    plt.suptitle(f"Timeline for {site_type} (model: {model})")
    plt.tight_layout()
    plt.show()

# Plot the histograms of transformed values and non-transformed values
def transformed_features_and_gpp_histograms(modelx, log=False, n_bins=50):
    X_test_tr = modelx.best_estimator_.regressor_.named_steps.scaler.transform(X_test)
    X_tv_tr = modelx.best_estimator_.regressor_.named_steps.scaler.transform(X_tv)
    nrows = 6
    feat = X_test.columns.values
    ncols = math.ceil(len(X_test.columns) / nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 12))
    i = 0
    for feat, ax in zip(feat, axes.flatten()):
        vm = min(X_tv_tr[:, i].min(), X_test_tr[:, i].min())
        vx = max(X_tv_tr[:, i].max(), X_test_tr[:, i].max())
        step = (vx - vm) / n_bins
        bins = np.arange(vm, vx, step)
        ax.hist(
            X_tv_tr[:, i],
            bins=bins,
            log=log,
            weights=np.ones_like(X_tv_tr[:, i]) / len(X_tv_tr[:, i]),
        )
        ax.hist(
            X_test_tr[:, i],
            bins=bins,
            log=log,
            alpha=0.8,
            weights=np.ones_like(X_test_tr[:, i]) / len(X_test_tr[:, i]),
        )
        ax.set_title(feat)
        i = i + 1

    plt.suptitle("Histograms of primary features")
    plt.tight_layout()
    plt.show()

    y_tv_tr = modelx.best_estimator_.transformer_.transform(y_tv)
    y_test_tr = modelx.best_estimator_.transformer_.transform(y_test)
    vm = min(y_tv_tr.min(), y_test_tr.min())
    vx = max(y_tv_tr.max(), y_test_tr.max())
    step = (vx - vm) / n_bins
    plt.hist(y_tv_tr, bins=bins, log=log, weights=np.ones_like(y_tv_tr) / len(y_tv_tr))
    plt.hist(
        y_test_tr,
        bins=bins,
        log=log,
        alpha=0.8,
        weights=np.ones_like(y_test_tr) / len(y_test_tr),
    )
    plt.show()

# Plot the characteristic of the transformer
def characteristic(model, y_true):
    y_test_tr = model.best_estimator_.transformer_.transform(
        pd.DataFrame(y_true)
    ).squeeze()
    # vm = min(df_pred["True GPP"].min(), y_test_tr.min())
    # vx = max(df_pred["True GPP"].max(), y_test_tr.max())
    # plt.plot([vm, vx], [vm, vx], c="gray")
    sns.scatterplot(
        x=y_true,
        y=y_test_tr,
        alpha=0.7,
        # hue=X_test[f].head(limit),
    )
    plt.ylabel("True GPP")
    plt.ylabel("Transformed GPP")
    plt.tight_layout()
    plt.show()

# Predict all models
def predict_all(X, y, label, models=model_list):
    results = []
    d = y.copy()
    for model_name in model_list:
        print(f"Predicting {model_name}...")
        model = reload_model(model_name)
        (
            d[model_name],
            pred_time,
            r2,
            mae,
            rmse,
        ) = compute_metrics(model, X, y, verbose=False)
        res = {
            "model": model_name,
            f"pred_time_{label}": pred_time,
            f"R2_{label}": r2,
            f"MAE_{label}": mae,
            f"RMSE_{label}": rmse,
        }
        results.append(res)
    d.rename(columns={"GPP": "True GPP"}, inplace=True)
    d_err = d.sub(d["True GPP"], axis=0)
    return d, d_err, pd.DataFrame(results).set_index("model")