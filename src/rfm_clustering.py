import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

data = pd.read_csv("data/cleaned_retail.csv", parse_dates=["InvoiceDate"])

# to compute recency im setting current (snapshot date) as last date in the dataset + 1 day
snapshot_date = data["InvoiceDate"].max() + pd.Timedelta(days=1)

# creating the rfm dataset
rfm = data.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (snapshot_date-x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalPrice", "sum")
).reset_index()

# scale recency frequency and monetary values so 1 doesn't dominate
rfm_log = np.log1p(rfm[["Recency", "Frequency", "Monetary"]])
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)

# start k-means implementation

# we gotta find an optimal k
wcss, sil_scores = [], []
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(rfm_scaled)
    wcss.append(km.inertia_)
    sil_scores.append(silhouette_score(rfm_scaled, labels))

# plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(2, 9), wcss, marker="o")
plt.title("Elbow Method"); plt.xlabel("k"); plt.ylabel("WCSS")
plt.subplot(1, 2, 2)
plt.plot(range(2, 9), sil_scores, marker="o", color="orange")
plt.title("Silhouette Score"); plt.xlabel("k"); plt.ylabel("Score")
plt.tight_layout()
plt.show()
print("Silhouette scores:", dict(zip(range(2, 9), sil_scores)))

# silhouette plot peaks at 2, too simple though so lets go with 4 clusters
k = 4
km = KMeans(n_clusters=k, random_state=42, n_init=10)
rfm["Cluster"] = km.fit_predict(rfm_scaled)
print(rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean().round(2))

rfm.to_csv("data/rfm_clusters.csv", index=False)
