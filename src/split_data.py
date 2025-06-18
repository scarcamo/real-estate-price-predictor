import os
import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_VARIABLE = "price"
RANDOM_STATE = 42



def _get_data() -> pd.DataFrame:
    """Loads and cleans the processed data."""
    base_path = os.path.join(os.path.dirname(__file__), "..", "data")
    df = pd.read_csv(os.path.join(base_path, "real_estate_thesis_processed.csv"), index_col=0, low_memory=False)

    # Load and merge image data
    img_data = pd.read_feather(os.path.join(base_path, "img_all_mean.feather"))
    img_data.set_index("id", inplace=True)
    img_data.drop(columns=[col for col in img_data.columns if 'Unnamed' in col], inplace=True)

    interior = img_data[img_data['classification'] == 'interior'].drop(columns=['classification'])
    exterior = img_data[img_data['classification'] == 'exterior'].drop(columns=['classification'])
    unfurnished_space = img_data[img_data['classification'] == 'unfurnished_space'].drop(columns=['classification'])
    other_interior = img_data[img_data['classification'] == 'other_interior'].drop(columns=['classification'])

    interior.columns = [col.replace("feature_", "interior_") for col in interior.columns]
    exterior.columns = [col.replace("feature_", "exterior_") for col in exterior.columns]
    unfurnished_space.columns = [col.replace("feature_", "unfurnished_space_") for col in unfurnished_space.columns]
    other_interior.columns = [col.replace("feature_", "other_interior_") for col in other_interior.columns]

    
    img_data = pd.concat([interior, exterior, unfurnished_space, other_interior], axis=1)

    # img dummy, prefix is img_cat_
    img_dummy = pd.read_csv(os.path.join(base_path, "img_cat.csv"))
    img_dummy.drop_duplicates(subset=["id"], inplace=True)
    img_dummy.set_index("id", inplace=True)

    # validate non null values in img_data
    if img_data.isnull().all().any():
        raise ValueError("img_data contains null values")

    df = df.merge(img_data, left_index=True, right_index=True, how="left")
    df = df.merge(img_dummy, left_index=True, right_index=True, how="left")

    # Check for presence of images per row
    interior_cols = df.filter(regex='interior_').columns
    unfurnished_space_cols = df.filter(regex='unfurnished_space_').columns
    
    # A row has images if it has at least one non-null value in any image category
    df['has_img'] = (
        (~df[interior_cols].isnull().all(axis=1)) |
        (~df[unfurnished_space_cols].isnull().all(axis=1))
    )

    # Remove outliers
    if 'outlier' in df.columns:
        print(f"Removing {df['outlier'].sum()} outliers from the dataset.")
        df = df[~df['outlier']]

    print(f"Found {df['has_img'].sum()} properties with images out of {len(df)} total")
    print(f"Removing {(~df['has_img']).sum()} properties without images from the dataset.")
    df = df[df['has_img']]

    return df


def stratified_split(df: pd.DataFrame, features: list[str], target: str, test_size: float = 0.2, random_state: int = RANDOM_STATE):
    """Performs stratified split based on price bins and has_img flag."""
    price_bins = [0, 250000, 500000, 1000000, 1500000, 2000000, 3000000, float("inf")]
    price_labels = [
        "0-250k",
        "250-500k",
        "500k-1M",
        "1M-1.5M",
        "1.5M-2M",
        "2M-3M",
        "3M+",
    ]

    features = [f for f in features if f != target]

    df_temp = df.copy()

    df_temp["price_bin"] = pd.cut(
        df_temp[target], bins=price_bins, labels=price_labels, right=False
    )

    # Create combined stratification column using both price bin and has_img
    df_temp["strat_combined"] = df_temp["price_bin"].astype(str) + "_" + df_temp["has_img"].astype(str)

    # Handle cases where some bins might be empty or have few samples
    counts = df_temp["strat_combined"].value_counts()
    min_samples_per_bin = 2  # Stratify needs at least 2 samples per class
    valid_bins = counts[counts >= min_samples_per_bin].index
    df_filtered = df_temp[df_temp["strat_combined"].isin(valid_bins)]

    if len(df_filtered) < len(df_temp):
        print(
            f"Warning: Removed {len(df_temp) - len(df_filtered)} samples due to insufficient samples in combined price/image bins for stratification."
        )

    X = df_filtered[features]
    Y = df_filtered[[target]]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        stratify=df_filtered["strat_combined"],
        random_state=random_state,
        test_size=test_size,
    )
    return X_train, X_test, Y_train, Y_test


def get_train_test_data(data: pd.DataFrame, include_img=True, include_count=True) -> tuple:
    """Splits the data into train and test sets."""
    X_train, X_test, y_train, y_test = stratified_split(
        data,
        data.columns.drop(TARGET_VARIABLE, errors='ignore'),
        TARGET_VARIABLE
    )

    assert X_train.index.name == "id", "X_train index name should be 'id'"
    assert X_test.index.name == "id", "X_test index name should be 'id'"
    assert y_train.index.name == "id", "y_train index name should be 'id'"
    assert y_test.index.name == "id", "y_test index name should be 'id'"

    if not include_img:
        # Drop image columns if they exist
        img_cols = X_train.filter(regex='img_|feature_|vector_|interior_|exterior_|unfurnished_space_|other_interior_').columns.tolist()
        X_train.drop(columns=img_cols, inplace=True, errors="raise")
        X_test.drop(columns=img_cols, inplace=True, errors="raise")

    if not include_count:
        # Drop count columns if they exist
        count_cols = X_train.filter(regex='count_').columns.tolist()
        X_train.drop(columns=count_cols, inplace=True, errors="raise")
        X_test.drop(columns=count_cols, inplace=True, errors="raise")

    # Separate image features
    img_cols = data.filter(regex='img_|feature_|vector_|interior_|exterior_|unfurnished_space_|other_interior_').columns.tolist()
    img_train = X_train[img_cols]
    img_test = X_test[img_cols]

    # drop img from X
    X_train.drop(columns=img_cols, inplace=True, errors="raise")
    X_test.drop(columns=img_cols, inplace=True, errors="raise")

    return X_train, X_test, y_train, y_test, img_train, img_test


def split_data(n_samples: int = None):
    base_path = os.path.join(os.path.dirname(__file__), "..", "data")

    df = _get_data()

    if n_samples:
        df = df.sample(n=n_samples, random_state=RANDOM_STATE)

    X_train, X_test, y_train, y_test, img_train, img_test = get_train_test_data(df)

    X_train.to_csv(os.path.join(base_path, "X_train.csv"), index=True)
    X_test.to_csv(os.path.join(base_path, "X_test.csv"), index=True)
    y_train.to_csv(os.path.join(base_path, "y_train.csv"), index=True)
    y_test.to_csv(os.path.join(base_path, "y_test.csv"), index=True)
    img_train.to_csv(os.path.join(base_path, "img_train.csv"), index=True)
    img_test.to_csv(os.path.join(base_path, "img_test.csv"), index=True)

    print(f"Split data into train and test sets with {len(X_train)} and {len(X_test)} samples respectively.")

if __name__ == "__main__":
    split_data()