import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

MAX_SEQ_LEN = 50
TEST_SIZE   = 0.2
SEED        = 42


def build_order_sequences(trans_path="data/cleaned_retail.csv",
                          cluster_path="data/rfm_clusters.csv"):
    transactions = pd.read_csv(trans_path, parse_dates=["InvoiceDate"])
    rfm_data     = pd.read_csv(cluster_path)

    order_features = transactions.groupby(["CustomerID", "InvoiceNo"]).agg(
        total_items     = ("Quantity",   "sum"),
        total_spend     = ("TotalPrice", "sum"),
        unique_products = ("StockCode",  "nunique"),
        avg_unit_price  = ("UnitPrice",  "mean"),
        order_time      = ("InvoiceDate", "min"),
    ).reset_index()

    order_features["day_of_week"] = order_features["order_time"].dt.dayofweek / 6.0
    order_features["hour"]        = order_features["order_time"].dt.hour / 23.0
    order_features = order_features.sort_values(
        ["CustomerID", "order_time"]
    ).reset_index(drop=True)

    order_features["days_since_prev"] = (
        order_features.groupby("CustomerID")["order_time"]
        .diff()
        .dt.total_seconds() / 86400.0
    ).fillna(0.0)

    feature_columns = [
        "total_items", "total_spend", "unique_products",
        "avg_unit_price", "day_of_week", "hour", "days_since_prev",
    ]

    customer_sequences, cluster_labels, customer_ids = [], [], []
    customer_to_cluster = dict(zip(rfm_data["CustomerID"], rfm_data["Cluster"]))

    for customer_id, customer_orders in order_features.groupby("CustomerID"):
        if customer_id not in customer_to_cluster:
            continue
        customer_sequences.append(customer_orders[feature_columns].values)
        cluster_labels.append(customer_to_cluster[customer_id])
        customer_ids.append(customer_id)

    return customer_sequences, np.array(cluster_labels), customer_ids, feature_columns


def pad_and_scale(sequences, labels, max_len, scaler=None, fit_scaler=False):
    seq_lengths = [len(seq) for seq in sequences]
    all_orders_flat = np.vstack(sequences)

    if fit_scaler:
        scaler = StandardScaler().fit(all_orders_flat)
    scaled_flat = scaler.transform(all_orders_flat)

    scaled_sequences, offset = [], 0
    for length in seq_lengths:
        scaled_sequences.append(scaled_flat[offset:offset + length])
        offset += length

    num_features = all_orders_flat.shape[1]
    padded_sequences = np.zeros((len(sequences), max_len, num_features), dtype=np.float32)
    padding_mask     = np.zeros((len(sequences), max_len),               dtype=np.float32)

    for i, seq in enumerate(scaled_sequences):
        seq_len = min(len(seq), max_len)
        # keep most recent orders (truncate from the front)
        trimmed = seq[-seq_len:]
        padded_sequences[i, :seq_len] = trimmed
        padding_mask[i, :seq_len]     = 1.0

    return padded_sequences, padding_mask, labels, scaler


def load_seq_splits(trans_path="data/cleaned_retail.csv",
                    cluster_path="data/rfm_clusters.csv",
                    max_len=MAX_SEQ_LEN, test_size=TEST_SIZE,
                    random_state=SEED):
    customer_sequences, cluster_labels, _, feature_columns = build_order_sequences(
        trans_path, cluster_path
    )

    all_indices = np.arange(len(customer_sequences))
    train_indices, test_indices = train_test_split(
        all_indices, test_size=test_size, random_state=random_state,
        stratify=cluster_labels
    )

    train_sequences = [customer_sequences[i] for i in train_indices]
    test_sequences  = [customer_sequences[i] for i in test_indices]
    train_labels, test_labels = cluster_labels[train_indices], cluster_labels[test_indices]

    train_padded, train_mask, train_labels, scaler = pad_and_scale(
        train_sequences, train_labels, max_len, fit_scaler=True
    )
    test_padded, test_mask, test_labels, _ = pad_and_scale(
        test_sequences, test_labels, max_len, scaler=scaler
    )

    return train_padded, train_mask, train_labels, test_padded, test_mask, test_labels, scaler


if __name__ == "__main__":
    print("Building order-level sequences …")
    customer_sequences, cluster_labels, customer_ids, feature_columns = build_order_sequences()
    print(f"  Customers : {len(customer_sequences)}")
    print(f"  Features  : {len(feature_columns)}  {feature_columns}")
    seq_lengths = [len(seq) for seq in customer_sequences]
    print(f"  Seq length: median={np.median(seq_lengths):.0f}, "
          f"mean={np.mean(seq_lengths):.1f}, max={np.max(seq_lengths)}")
    print(f"  Clusters  : {np.bincount(cluster_labels)}")

    print(f"\nSplitting and scaling (max_len={MAX_SEQ_LEN}) …")
    (train_padded, train_mask, train_labels,
     test_padded, test_mask, test_labels, scaler) = load_seq_splits()

    print(f"  Train : {train_padded.shape}  Test : {test_padded.shape}")

    np.save("data/seq_X_train.npy",    train_padded)
    np.save("data/seq_X_test.npy",     test_padded)
    np.save("data/seq_mask_train.npy", train_mask)
    np.save("data/seq_mask_test.npy",  test_mask)
    np.save("data/seq_y_train.npy",    train_labels)
    np.save("data/seq_y_test.npy",     test_labels)
    joblib.dump(scaler, "models/seq_scaler.pkl")
    print("  Saved to data/seq_*.npy + models/seq_scaler.pkl")