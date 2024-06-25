import os
import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt

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

# List non-QC features
def features(data):
    return [f for f in data.columns.values if not f.endswith("_QC")]  
    
def build_smooth_discard_mask(df, bad_data_mask):
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
        return smooth_bad_data_mask
        
def build_windowed_discard_mask(df, bad_data_mask, window_size, na_check_columns):
    def windowed_mask(d, m, mask_name="mask"):
        def g(x, n):
            return x.sum() == n

        def r(d, n):
            return d[mask_name].rolling(n, step=n).apply(lambda x: g(x, n))

        windowed_mask_name = mask_name + "_window"
        d[windowed_mask_name] = r(d, m)
        d[windowed_mask_name] = d[windowed_mask_name].shift(1 - m)
        d[windowed_mask_name].ffill(limit=m - 1, inplace=True)
        return d
    
    a = pd.DataFrame()
    a["mask"] = (~bad_data_mask) & (~df[na_check_columns].isnull().any(axis=1))
    a = a.groupby("site").apply(lambda x: windowed_mask(x.droplevel(0), window_size))
    return (a["mask_window"] != 1)

        
def add_rolling_window_features(df, rolling_variables):
    print("Adding rolling window features")

    def func(x, window):
        df = x.loc[x.index[0][0]]
        return df.rolling(window=f"{window}d", min_periods=int(window / 2)).apply(
            lambda w: w.mean()
        )

    for f in rolling_variables:
        for period in [1, 4]:
            print(f"Processing {f}_{period}w...")
            df[f"{f}_{period}w"] = df.groupby("site").apply(
                lambda x: func(x[f], period * 7)
            )
    
# Load data from path and performs the processing defined in the EDA (feature engineering, bad removal).
# It does not perform scaling or outlier remval so there is no data leakage here.
# If qc_threshold is defined (between 0 and 1), cleaning badsed on QC values is performed.
# If window_size is set, full well-formed windows are kept and rolling variables are not added (the window_size parameter does not configure the rolling variables)
# For details about the actions performed, please see in the EDA.
def process_multi_site(path, metadata_path, qc_threshold=None, show_na=False, keep_qc=False, rolling_variables=["P_F", "TA_F_MDS", "VPD_F_MDS"], window_size=None, interpolate=False):
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

    if qc_threshold:
        qc_check=["TA_F_MDS_QC", "SW_IN_F_MDS_QC", "VPD_F_MDS_QC", "NEE_VUT_REF_QC"]
        bad_data_mask = (df[qc_check] < qc_threshold).any(axis=1)
        if window_size is None:
            discard_mask = build_smooth_discard_mask(df, bad_data_mask)
        else:
            na_check_columns = [ "TA_F_MDS", "GPP" ]
            discard_mask = build_windowed_discard_mask(df, bad_data_mask, window_size, na_check_columns)
        df = df.copy()[~discard_mask]
    
    add_rolling_window_features(df, rolling_variables)

    print("Adding engineered features")
    df["day_length"] = df.apply(lambda x: daylength(x["day_of_year"], x["lat"]), axis=1)
    df["solar_altitude"] = df.apply(
        lambda x: solar_altitude(x["day_of_year"], x["lat"]), axis=1
    )
    df["apar"] = df["SW_IN_F_MDS"] * df["FPAR"]
    df["TA_F_MDS**2"] = (df["TA_F_MDS"] + 41) ** 2
    try:
        df["TA_F_MDS_1w**2"] = (df["TA_F_MDS_1w"] + 41) ** 2
        df["TA_F_MDS_4w**2"] = (df["TA_F_MDS_4w"] + 41) ** 2
    except:
        pass

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
        "VPD_DAY_F_MDS",
        "TA_DAY_F_MDS"
    ]
    if window_size is None:
        excluded_features = excluded_features + ["P_F"]
    if not keep_qc:
        excluded_features = excluded_features + ["GPP_diff"]
    working_features = [f for f in df.columns.values if (f not in excluded_features) and (keep_qc or (not f.endswith("_QC")))]
    df = df[working_features].copy()
    
    print(f"NA: {(df.isna().any(axis=1)).sum()}")

    if interpolate is True or window_size is not None:
        print("Interpolating")
        df.interpolate(inplace=True)
    
    if show_na:
        msno.matrix(df, labels=True)
        plt.show()

    return df


# Load data from path and performs the processing defined in the EDA (feature engineering, bad removal).
# It does not perform scaling or outlier remval so there is no data leakage here.
# For details about the actions performed, please see in the EDA.
def process_single_site(path, metadata_path, with_rolling_windows=True, interpolate=False):
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

    if interpolate:
        print("Interpolating")
        df.interpolate(inplace=True)
        
    msno.matrix(df, labels=True)
    plt.show()
    
    return df
