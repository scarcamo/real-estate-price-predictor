import os
import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_VARIABLE = "price"
N_SAMPLES = None
RANDOM_STATE = 42


def stratified_split(df, features, target, test_size=0.2, random_state=RANDOM_STATE):
    """Performs stratified split based on price bins."""
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

    df_temp = df_temp[~df_temp['outlier']]

    print(f"Removed {len(df) - len(df_temp)} outliers from the dataset.")

    df_temp["price_bin"] = pd.cut(
        df_temp[target], bins=price_bins, labels=price_labels, right=False
    )

    # Handle cases where some bins might be empty or have few samples
    counts = df_temp["price_bin"].value_counts()
    min_samples_per_bin = 2  # Stratify needs at least 2 samples per class
    valid_bins = counts[counts >= min_samples_per_bin].index
    df_filtered = df_temp[df_temp["price_bin"].isin(valid_bins)]

    if len(df_filtered) < len(df_temp):
        print(
            f"Warning: Removed {len(df_temp) - len(df_filtered)} samples due to insufficient samples in price bins for stratification."
        )

    X = df_filtered[features]
    Y = df_filtered[target]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        stratify=df_filtered["price_bin"],
        random_state=random_state,
        test_size=test_size,
    )
    return X_train, X_test, Y_train, Y_test


def get_train_test_data(include_img=True, include_count=True) -> tuple:
    """Loads the data and splits it into train and test sets."""
    X_train = pd.read_csv(os.path.join("data", "X_train.csv"), index_col=0, low_memory=False)
    X_test = pd.read_csv(os.path.join("data", "X_test.csv"), index_col=0, low_memory=False)
    y_train = pd.read_csv(os.path.join("data", "y_train.csv"), index_col=0, low_memory=False)
    y_test = pd.read_csv(os.path.join("data", "y_test.csv"), index_col=0, low_memory=False)

    assert X_train.index.name == "id", "X_train index name should be 'id'"
    assert X_test.index.name == "id", "X_test index name should be 'id'"
    assert y_train.index.name == "id", "y_train index name should be 'id'"
    assert y_test.index.name == "id", "y_test index name should be 'id'"

    if not include_img:
        # Drop image columns if they exist
        img_cols = X_train.filter(regex='img_|feature_|vector_|interior_|exterior_').columns.tolist()
        X_train.drop(columns=img_cols, inplace=True, errors="raise")
        X_test.drop(columns=img_cols, inplace=True, errors="raise")

    if not include_count:
        # Drop count columns if they exist
        count_cols = X_train.filter(regex='count_').columns.tolist()
        X_train.drop(columns=count_cols, inplace=True, errors="raise")
        X_test.drop(columns=count_cols, inplace=True, errors="raise")

    return X_train, X_test, y_train, y_test


def get_train_test_img() -> tuple:
    """Loads the data and splits it into train and test sets."""
    img_train = pd.read_csv(os.path.join("data", "img_train.csv"), index_col=0)
    img_test = pd.read_csv(os.path.join("data", "img_test.csv"), index_col=0)

    assert img_train.index.name == "id", "img_train index name should be 'id'"
    assert img_test.index.name == "id", "img_test index name should be 'id'"

    return img_train, img_test


if __name__ == "__main__":
    df = pd.read_csv(
        os.path.join("data", "real_estate_thesis_processed.csv"), index_col=0
    )

    img_data = pd.read_csv(os.path.join("data", "img_interior_max.csv"))
    img_data.set_index("id", inplace=True)

    df = df.merge(img_data, left_index=True, right_index=True, how="left")

    if N_SAMPLES:
        df = df.sample(n=N_SAMPLES, random_state=RANDOM_STATE)

    X_train, X_test, y_train, y_test = stratified_split(
        df,
        df.columns.drop(TARGET_VARIABLE),
        TARGET_VARIABLE,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    X_train.to_csv(os.path.join("data", "X_train.csv"), index=True)
    X_test.to_csv(os.path.join("data", "X_test.csv"), index=True)
    y_train.to_csv(os.path.join("data", "y_train.csv"), index=True)
    y_test.to_csv(os.path.join("data", "y_test.csv"), index=True)

    if X_train.index.name != "id" or X_test.index.name != "id":
        raise ValueError("X_train and X_test index name should be 'id'")

    train_ids = X_train.index.tolist()
    test_ids = X_test.index.tolist()

    img_train = img_data[img_data.index.isin(train_ids)]
    img_test = img_data[img_data.index.isin(test_ids)]

    img_train.to_csv(os.path.join("data", "img_train.csv"), index=True)
    img_test.to_csv(os.path.join("data", "img_test.csv"), index=True)
