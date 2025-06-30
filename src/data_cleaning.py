import os
import re

import numpy as np
import pandas as pd
from unidecode import unidecode

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def clean_text(text_series: pd.Series):
    """
    Clean and normalize a pandas Series of text for OHE.
    Expects a single pandas Series as input.
    Converts NaN/None values to the string 'missing_value'.
    """
    if not isinstance(text_series, pd.Series):
        if isinstance(text_series, (np.ndarray, list)) and np.ndim(text_series) == 1:
            text_series = pd.Series(text_series)
        else:
            raise TypeError(
                f"clean_text expected a pandas Series or 1D input, but got {type(text_series)} with ndim={getattr(text_series, 'ndim', 'N/A')}"
            )

    def clean_single_text_value(text):
        """Internal helper to clean and normalize a single text value."""
        if pd.isna(text):
            return "missing_value"
        text = str(text).lower()
        text = unidecode(text)
        text = re.sub(r"[^a-z0-9_]+", "_", text)
        text = text.strip("_")
        return text if text else "empty_value"

    return text_series.apply(clean_single_text_value).astype(str)


def clean_data():
    df = pd.read_csv(os.path.join("data", "real_estate_thesis_pois.csv"), low_memory=False)


    df.drop(columns=["Unnamed: 0"], inplace=True)


    df["date_created"] = pd.to_datetime(df["date_created"])
    df["date_created_first"] = pd.to_datetime(df["date_created_first"])
    df["pushed_up_at"] = pd.to_datetime(df["pushed_up_at"], errors="coerce", utc=True)


    df["price_per_m2"] = df["price"] / df["area_m2"]


    df.drop_duplicates(subset=["id"], inplace=True)

    df.set_index("id", inplace=True)


    # we only have SELL type data
    df = df[df["transaction_type"] == "SELL"]

    df.rename(
        columns={
            "location.district": "district",
            "location.latitude": "latitude",
            "location.longitude": "longitude",
            "location.neighborhood": "neighborhood",
        },
        inplace=True,
    )


    # remove outliers, where price=1, etc
    df = df[df["estate"] == "FLAT"]

    df.drop(
        columns=[
            "estate",
            "transaction_type",
            "date_created",
            "date_created_first",
            "pushed_up_at",
            "count",
            "geometry",
        ],
        inplace=True,
    )


    # filter out
    cond1 = df["price_per_m2"] > 2000
    cond2 = df["area_m2"] >= 10

    df = df[cond1 & cond2]

    # Add panorama data
    panorama = pd.read_csv(os.path.join("data", "panorama_districts.csv"), index_col=0)
    panorama.reset_index(inplace=True)
    # there are districts with 0 values in some cases such as cemeteries, parks, etc.
    panorama.fillna(0, inplace=True)
    panorama["pano_share_pop_females"] = (
        panorama["pano_pop_females"] / panorama["pano_pop_total"]
    )


    df = df.reset_index()
    df = df.merge(panorama, on="district", how="left")
    df.set_index("id", inplace=True)


    # Data transformation

    floor_mapping = {
        "CELLAR": -1,
        "GROUND": 0,
        "FIRST": 1,
        "SECOND": 2,
        "THIRD": 3,
        "FOURTH": 4,
        "FIFTH": 5,
        "SIXTH": 6,
        "SEVENTH": 7,
        "EIGHTH": 8,
        "NINTH": 9,
        "TENTH": 10,
        "ABOVE_TENTH": 11,
        "GARRET": 12,
    }


    df["floor_number"] = df["floor_number"].map(floor_mapping)

    df["building_year"] = np.select(
        [df["building_year"] < 1800, df["building_year"] > 2028],
        [np.nan, np.nan],
        default=df["building_year"],
    )

    df["building_age"] = pd.Timestamp.now().year - df["building_year"]

    city_center = (52.2318543, 21.0028622)  # for Warsaw

    # 1 degree of latitude/longitude ≈ 111 km
    lat_diff = df["latitude"] - city_center[0]
    lon_diff = df["longitude"] - city_center[1]
    df["distance_to_center_km"] = np.sqrt(lat_diff**2 + lon_diff**2) * 111

    df["room_size_m2"] = df["area_m2"] / df["rooms_number"].replace("MORE", 11).astype(
        float
    )
    df["floor_ratio"] = df["floor_number"] / df["building_floors_num"]


    # select most common values for categorical variables
    top_heating = df["heating"].value_counts().index[:4]
    print("Top heating types:", top_heating)
    df["heating"] = df["heating"].apply(lambda x: x if x in top_heating else "OTHER")

    top_building_type = df["building_type"].value_counts().index[:3]
    excluded_building_types = [ bt for bt in df["building_type"].unique() if bt not in top_building_type ]
    print("Top building types:", top_building_type)
    print("Excluding", excluded_building_types)
    df["building_type"] = df["building_type"].apply(
        lambda x: x if x in top_building_type else "OTHER"
    )


    df["building_material"] = df["building_material"].replace("[]", "MISSING")

    top_building_material = df["building_material"].value_counts().index[:6]
    print("Top building materials:", top_building_material)
    df["building_material"] = df["building_material"].apply(
        lambda x: x if x in top_building_material else "OTHER"
    )


    df["rooms_number"] = (
        df["rooms_number"].astype("str").replace("MORE", "11").astype("Int64")
    )

    df["is_private_owner"] = df["is_private_owner"].astype(int)

    ### remove outliers
    multiplier = 1.5
    columns = ["area_m2", "price", "building_floors_num", "price_per_m2", "room_size_m2"]

    df["outlier"] = False  # Initialize all as non-outliers

    for col in columns:
        if col in ["price", "price_per_m2"]:
            values = np.log(df[col])
        else:
            values = df[col]

        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        if col in ["price", "price_per_m2"]:
            mask = (np.log(df[col]) < lower_bound) | (np.log(df[col]) > upper_bound)
        else:
            mask = (df[col] < lower_bound) | (df[col] > upper_bound)

        df.loc[mask, "outlier"] = True  # Mark outliers

    print(df["outlier"].sum(), "rows marked as outliers.")


    # remove columns with only one unique val and those that are redundant
    cols_to_drop = ["park_areas_2000_m2", "park_areas_1000_m2"]
    for col in df.columns:
        if len(df[col].unique()) == 1:
            cols_to_drop.append(col)

    df.drop(columns=cols_to_drop, inplace=True)
    print("Removed columns", str(cols_to_drop), "with one unique value")


    # feature engineering

    df["has_street_name"] = df["location.name"].notna()
    df["has_street_number"] = df["location.number"].notna()

    if "building_year" in df.columns:
        df.drop(columns=["building_year"], inplace=True)


    df.drop(
        columns=[
            "location.name",
            "location.number",
            # "neighborhood",
            "has_street_name",
            "has_street_number",
            "price_per_m2",
            "latitude",
            "longitude",
        ],
        inplace=True,
    )


    # Clean categorical columns

    categorical_cols = df.select_dtypes(include=[object, "category"]).columns.tolist()

    for col in categorical_cols:
        if col in df.columns:
            cleaned_series = clean_text(df[col])
            df[col] = cleaned_series.astype("category")


    # drop properties outside of warsaw
    filename = "data/invalid_properties_indexes.txt"

    loaded_index = []

    with open(filename, "r") as f:
        for line in f:
            loaded_index.append(int(line.strip()))

    invalid_properties_indexes = pd.Index(loaded_index)


    indices_to_drop = [
        65349901,
        65949392,
        66357797,
        65818114,
    ]


    df = df.drop(indices_to_drop, axis=0)
    df = df.drop(invalid_properties_indexes, axis=0, errors='ignore')

    df.columns = [col.replace("—", "-") for col in df.columns]

    # drop vector columns
    vector_columns = [col for col in df.columns if col.startswith("vector_")]
    df.drop(columns=vector_columns, inplace=True)

    df.to_feather(os.path.join("data", "real_estate_thesis_processed.feather"))

if __name__ == "__main__":
    clean_data()