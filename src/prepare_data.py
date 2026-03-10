import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_rfm_splits(csv_path="data/rfm_clusters.csv", test_size=0.2,
                    random_state=42):
    """Load RFM data, preprocess, and return train/test splits as numpy arrays."""
    rfm = pd.read_csv(csv_path)

    features = rfm[["Recency", "Frequency", "Monetary"]].values
    labels = rfm["Cluster"].values

    # log-transform then standardize (same preprocessing as clustering step)
    features_log = np.log1p(features)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_log)

    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, labels,
        test_size=test_size, random_state=random_state, stratify=labels,
    )

    return X_train, X_test, y_train, y_test
