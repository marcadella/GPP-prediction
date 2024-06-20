import h5py
import io
from tensorflow import keras
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras import regularizers, layers, models, optimizers, callbacks
from sklearn.base import BaseEstimator
from pickle import load
import os
import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import time
from sklearn.metrics import (
    mean_absolute_error as MAE,
    mean_squared_error as RMSE,
    r2_score as R2,
)

# KerasRegressor is not pickleable. This wrapper is meant to solve this issue, but it does work with callbacks...
# Solution from https://github.com/keras-team/keras/issues/13168#issuecomment-672792106
class PickleableKerasRegressor(KerasRegressor, BaseEstimator):       
    def _pickle_model(self):
        bio = io.BytesIO()
        with h5py.File(bio, "w") as f:
            self.model.save(f)
        return bio

    def _unpickle_model(self, model):
        with h5py.File(model, "r") as f:
            model = models.load_model(f)
        return model

    def __getstate__(self):
        state = BaseEstimator.__getstate__(self)
        if hasattr(self, "model"):
            state["model"] = self._pickle_model()
        return state

    def __setstate__(self, state):
        if state.get("model", None):
            state["model"] = self._unpickle_model(state["model"])
        return BaseEstimator.__setstate__(self, state)


# It is unfortunately not possible to get the history when using KerasRegressor within a Pipeline.
# This wrapper is solving this issue by saving the history locally.
# It is however not pickleable somehow, so we could not merge it with PickleableKerasRegressor.
class HistoryKerasRegressor(KerasRegressor, BaseEstimator):
    def __init__(self, *args, **kwargs):
        super(KerasRegressor, self).__init__(*args, **kwargs)
        self.history = None
        
    def get_history(self):
        return self.history
    
    def fit(self, X, y, *args, **kwargs):
        self.history = KerasRegressor.fit(self, X, y, *args, **kwargs)
        return self.history
        

# Function generating our Keras model
def model_fun(
    hidden=3,
    neurons=30,
    dropout=0,
    learning_rate=0.003,
    input_shape=[22],
    activation="relu",
    l2=0.01,
    loss="mse",
    show_summary=False
):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    for layer in range(hidden):
        model.add(
            layers.Dense(
                neurons, activation=activation, kernel_regularizer=regularizers.L2(l2)
            )
        )
        if dropout != 0:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1))
    optimizer = optimizers.SGD(learning_rate=learning_rate)
    model.compile(loss=loss, optimizer=optimizer)
    if show_summary:
        model.summary()
    return model


# Load a model by unpickleing it
# Do NOT call load_model because of name conflict!
def reload_model(name):
    with open(os.path.join("resources", f"{name}.pkl"), "rb") as f:
        model = load(f)
    return model

# Computes the declination from the day of the year
# Source: https://gist.github.com/anttilipp/ed3ab35258c7636d87de6499475301ce
def declination(dayOfYear):
    return 23.45 * np.sin(np.deg2rad(360.0 * (283.0 + dayOfYear) / 365.0))

# Computes the solar altitude from the day of the year and the latitude
# Source: https://gist.github.com/anttilipp/ed3ab35258c7636d87de6499475301ce
def solar_altitude(dayOfYear, lat):
    return max(90 - abs(lat - declination(dayOfYear)), 0.0)

# Computes the daylength from the day of the year and the latitude
# Source: https://gist.github.com/anttilipp/ed3ab35258c7636d87de6499475301ce
def daylength(dayOfYear, lat):
    latInRad = np.deg2rad(lat)
    declinationOfEarth = declination(dayOfYear)
    if -np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth)) <= -1.0:
        return 24.0
    elif -np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth)) >= 1.0:
        return 0.0
    else:
        hourAngle = np.rad2deg(
            np.arccos(-np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth)))
        )
        return 2.0 * hourAngle / 15.0

# Load data from path and performs the processing defined in the EDA (feature engineering, bad removal).
# It does not perform scaling or outlier remval so there is no data leakage here.
# If qc_threshold is defined (between 0 and 1), cleaning badsed on QC values is performed.
# For details about the actions performed, please see in the EDA.
def process_multi_site(path, metadata_path, qc_threshold=None, show_na=False, keep_qc=False, rolling_variables=["P_F", "TA_F_MDS", "VPD_F_MDS"]):
    print("Loading the data")
    df = pd.read_csv(path)
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    df["month"] = df["TIMESTAMP"].dt.month
    df["year"] = df["TIMESTAMP"].dt.year
    df["day_of_year"] = df["TIMESTAMP"].dt.dayofyear

    print("Loading metadata and joining")
    site_df = pd.read_csv(metadata_path, index_col="sitename")
    df_join = df.join(site_df[["lat"]], on="sitename")
    df["lat"] = df_join["lat"]

    df.set_index(["sitename", "TIMESTAMP"], drop=True, inplace=True)
    df.index.names = ["site", "date"]

    print("Adding GPP and GPP_diff")
    df["GPP"] = df[["GPP_NT_VUT_REF", "GPP_DT_VUT_REF"]].mean(axis=1, skipna=False)
    df["GPP_diff"] = df["GPP_NT_VUT_REF"] - df["GPP_DT_VUT_REF"]

    def features(data):
        return [f for f in data.columns.values if not f.endswith("_QC") or keep_qc]

    if qc_threshold:
        f_qc_check = ["TA_F_MDS_QC", "SW_IN_F_MDS_QC", "VPD_F_MDS_QC", "NEE_VUT_REF_QC"]
        bad_data_mask = (df[f_qc_check] < qc_threshold).any(axis=1)

        def blur(x):
            d = x.loc[x.index[0][0]]
            return d.rolling(window="29d", center=True, min_periods=14).apply(
                lambda w: w.sum() > len(w) / 2
            )

        smooth_bad_data_mask = (
            bad_data_mask.groupby("site").apply(lambda x: blur(x)) == 1
        )
        trash_count = smooth_bad_data_mask.sum()
        print(
            f"Samples to discard: {trash_count} ({(smooth_bad_data_mask.sum() / len(df)).round(2) * 100}%)"
        )
        df = df[features(df)].copy()[~smooth_bad_data_mask]

    print("Dropping unused features")
    discarded_features = [
        "NETRAD",
        "SW_OUT",
        "USTAR",
        "NEE_VUT_REF",
        "RECO_NT_VUT_REF",
        "H_CORR",
        "LE_CORR",
        "H_CORR_JOINTUNC",
        "LE_CORR_JOINTUNC",
        "TMIN_F_MDS",
        "TMAX_F_MDS",
    ]
    df.drop(discarded_features, inplace=True, axis=1)

    print("Removing bad values")

    def nullate_lower_than(d, field, limit=-9998):
        d.loc[d[field] <= limit, field] = np.nan

    for f in df.select_dtypes(include="number").columns:
        nullate_lower_than(df, f)

    print("Adding rolling window features")

    def func(x, window):
        d = x.loc[x.index[0][0]]
        return d.rolling(window=f"{window}d", min_periods=int(window / 2)).apply(
            lambda w: w.mean()
        )

    for f in rolling_variables:
        for period in [1, 4]:
            print(f"Processing {f}_{period}w...")
            df[f"{f}_{period}w"] = df.groupby("site").apply(
                lambda x: func(x[f], period * 7)
            )

    print("Adding engineered features")
    df["day_length"] = df.apply(lambda x: daylength(x["day_of_year"], x["lat"]), axis=1)
    df["solar_altitude"] = df.apply(
        lambda x: solar_altitude(x["day_of_year"], x["lat"]), axis=1
    )
    df["apar"] = df["SW_IN_F_MDS"] * df["FPAR"]
    df["TA_F_MDS**2"] = (df["TA_F_MDS"] + 41) ** 2
    df["TA_F_MDS_1w**2"] = (df["TA_F_MDS_1w"] + 41) ** 2
    df["TA_F_MDS_4w**2"] = (df["TA_F_MDS_4w"] + 41) ** 2

    print("Remove unused features")
    excluded_features = [
        "CO2_F_MDS",  # As explained above
        "GPP_DT_VUT_REF",  # Already included in GPP
        "GPP_NT_VUT_REF",  # Already included in GPP
        "TIMESTAMP",
        "sitename",
        "lat",
        "day_of_year",
        "month",
        "year",
    ]
    if not keep_qc:
        excluded_features = excluded_features + ["GPP_diff"]
    working_features = [f for f in features(df) if f not in excluded_features]
    df = df[working_features].copy()
    df.shape

    df.drop(["VPD_DAY_F_MDS", "TA_DAY_F_MDS", "P_F"], axis=1, inplace=True)
    df.shape

    print(f"NA: {(df.isna().any(axis=1)).sum()}")
    if show_na:
        msno.matrix(df, labels=True)
        plt.show()

    return df


# Load data from path and performs the processing defined in the EDA (feature engineering, bad removal).
# It does not perform scaling or outlier remval so there is no data leakage here.
# For details about the actions performed, please see in the EDA.
def process_single_site(path, metadata_path, with_rolling_windows=True):
    print("Loading the data")
    df = pd.read_csv(path)
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    df["month"] = df["TIMESTAMP"].dt.month
    df["year"] = df["TIMESTAMP"].dt.year
    df["day_of_year"] = df["TIMESTAMP"].dt.dayofyear
    df.set_index(["TIMESTAMP"], drop=True, inplace=True)
    df.index.names = ["date"]

    print("Loading metadata and joining")
    site_df = pd.read_csv(
        metadata_path, index_col="sitename"
    )
    df_join = df.join(site_df[["lat"]], on="sitename")
    df["lat"] = df_join["lat"]
    
    print("Adding GPP and GPP_diff")
    df["GPP"] = df[["GPP_NT_VUT_REF", "GPP_DT_VUT_REF"]].mean(axis=1, skipna=False)
    df["GPP_diff"] = df["GPP_NT_VUT_REF"] - df["GPP_DT_VUT_REF"]


    def features(data):
        return [f for f in data.columns.values if not f.endswith("_QC")]

    print("Dropping unused features")
    discarded_features = [
        "NETRAD",
        "SW_OUT",
        "USTAR",
        "NEE_VUT_REF",
        "RECO_NT_VUT_REF",
        "H_CORR",
        "LE_CORR",
        "H_CORR_JOINTUNC",
        "LE_CORR_JOINTUNC",
        "NETRAD_QC",
        "USTAR_QC",
        "H_CORR_QC",
        "LE_CORR_QC",
        "TMIN_F_MDS",
        "TMAX_F_MDS",
    ]
    df.drop(discarded_features, inplace=True, axis=1)

    print("Removing bad values")
    def nullate_lower_than(d, field, limit=-9998):
        d.loc[d[field] <= limit, field] = np.nan
        
    for f in df.select_dtypes(include="number").columns:
        nullate_lower_than(df, f)


    if with_rolling_windows:
        print("Adding rolling window features")
        def func(x, window):
            d = x.loc[x.index[0][0]]
            return d.rolling(window=f"{window}d", min_periods=int(window / 2)).apply(
                lambda w: w.mean()
            )


        for f in ["P_F", "TA_F_MDS", "VPD_F_MDS"]:
            for period in [1, 4]:
                print(f"Processing {f}_{period}w...")
                window = period * 7
                df[f"{f}_{period}w"] = (
                    df[f]
                    .rolling(window=f"{window}d", min_periods=int(window / 2))
                    .apply(lambda w: w.mean())
                )

    print("Adding engineered features")
    df["day_length"] = df.apply(lambda x: daylength(x["day_of_year"], x["lat"]), axis=1)
    df["solar_altitude"] = df.apply(
        lambda x: solar_altitude(x["day_of_year"], x["lat"]), axis=1
    )
    df["apar"] = df["SW_IN_F_MDS"] * df["FPAR"]
    df["TA_F_MDS**2"] = (df["TA_F_MDS"] + 41) ** 2
    if with_rolling_windows:
        df["TA_F_MDS_1w**2"] = (df["TA_F_MDS_1w"] + 41) ** 2
        df["TA_F_MDS_4w**2"] = (df["TA_F_MDS_4w"] + 41) ** 2

    print("Remove unused features")
    excluded_features = [
        "CO2_F_MDS",  # As explained above
        "GPP_DT_VUT_REF",  # Already included in GPP
        "GPP_NT_VUT_REF",  # Already included in GPP
        "GPP_diff",
        "TIMESTAMP",
        "sitename",
        "lat",
        "day_of_year",
        "month",
        "year",
    ]
    working_features = [f for f in features(df) if f not in excluded_features]
    df = df[working_features].copy()
    df.shape
    df.drop(["VPD_DAY_F_MDS", "TA_DAY_F_MDS"], axis=1, inplace=True)
    if with_rolling_windows:
        # With rolling windows, we exclude P_F as the precipitation happening today is unlikely to affect today's GPP
        df.drop(["P_F"], axis=1, inplace=True)
    df.shape

    print(f"NA: {(df.isna().any(axis=1)).sum()}")
    msno.matrix(df, labels=True)
    plt.show()
    
    return df


# Given a model and data, computes several metrics
def compute_metrics(model, X, y, label="", verbose=True):
    if verbose:
        print("----")
        print(f"Predicting {label}...")
    start = time.time()
    y_pred = model.predict(X)
    end = time.time()
    pred_time = round(end - start, 2)
    # Compute the metrics
    r2 = R2(y, y_pred).round(2)
    mae = MAE(y, y_pred).round(2)
    rmse = RMSE(y, y_pred).round(2)
    # Print
    if verbose:
        print(f"Prediction time: {pred_time}s")
        print(f"Score ({label}): {r2}")
        print(f"MAE ({label}): {mae}")
        print(f"RMSE ({label}): {rmse}")
    return y_pred, pred_time, r2, mae, rmse
