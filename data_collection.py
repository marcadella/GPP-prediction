import re
import tarfile
import pandas as pd

# Performs GPP presence check and add sitename variable:
def process_dataset(d, site_name):
    # Returns None if at least one of the two GPP features is entierly missingselected_site
    if (
        d["GPP_DT_VUT_REF"].isna().values.all()
        | d["GPP_NT_VUT_REF"].isna().values.all()
    ):
        print(f"Discarding {site_name} because all GPP values are missing.")
        return None
    # Add the sitename as a feature
    d["sitename"] = site_name
    return d


# Extract site name from file name
def extract_site_name(p):
    return re.findall("^FLX_([-a-zA-Z0-9]+)_", p)[0]


# Loading time series
def load_time_series(archive_path, sites, igbp_classes):
    # Open the archive
    with tarfile.open(archive_path, "r:*") as tar:
        # List all files
        csv_paths = tar.getnames()
        # Parse CSV files which belong to the selected land cover classes, and concatenate them in one dataframe,
        # applying process_dataset() to apply further filtering criteria
        df = pd.concat(
            [
                process_dataset(pd.read_csv(tar.extractfile(p)), extract_site_name(p))
                for p in csv_paths
                if sites.loc[extract_site_name(p), "igbp_land_use"] in igbp_classes
            ]
        )
    site_count = len(df["sitename"].unique())
    print()
    print(f"Loaded {site_count} sites ({len(df)} samples).")
    return df