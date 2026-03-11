import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

rfm = pd.read_csv("data/rfm_clusters.csv")

cluster_colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]
cluster_labels = {c: f"Cluster {c}" for c in range(4)}
n_clusters = rfm["Cluster"].nunique()

features = ["Recency", "Frequency", "Monetary"]
pairs = [("Recency", "Frequency"), ("Recency",
                                    "Monetary"), ("Frequency", "Monetary")]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (fx, fy) in zip(axes, pairs):
    for c in range(n_clusters):
        mask = rfm["Cluster"] == c
        ax.scatter(
            rfm.loc[mask, fx], rfm.loc[mask, fy],
            c=cluster_colors[c], label=cluster_labels[c],
            alpha=0.5, s=15, edgecolors="none"
        )
    ax.set_xlabel(fx)
    ax.set_ylabel(fy)
    ax.set_title(f"{fx} vs {fy}")
    ax.legend(fontsize=8, markerscale=2)
fig.suptitle("Pairwise Feature Scatter Plots by Cluster", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("src/figures/cluster_scatter_pairs.png",
            dpi=150, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, feat in zip(axes, features):
    data_by_cluster = [rfm.loc[rfm["Cluster"] == c, feat]
                       for c in range(n_clusters)]
    bp = ax.boxplot(data_by_cluster, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], cluster_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels([cluster_labels[c]
                       for c in range(n_clusters)], fontsize=8, rotation=15)
    ax.set_ylabel(feat)
    ax.set_title(f"{feat} by Cluster")
fig.suptitle("Feature Distributions by Cluster", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("src/figures/cluster_boxplots.png", dpi=150, bbox_inches="tight")
plt.show()

cluster_means = rfm.groupby("Cluster")[features].mean()
# normalize each feature to [0, 1] for comparison
cluster_norm = (cluster_means - cluster_means.min()) / \
    (cluster_means.max() - cluster_means.min())

x = np.arange(len(features))
width = 0.18
fig, ax = plt.subplots(figsize=(10, 6))
for c in range(n_clusters):
    ax.bar(x + c * width, cluster_norm.loc[c], width,
           color=cluster_colors[c], label=cluster_labels[c], edgecolor="white")
ax.set_xticks(x + width * (n_clusters - 1) / 2)
ax.set_xticklabels(features, fontsize=12)
ax.set_ylabel("Normalized Mean (0–1)")
ax.set_title("Cluster Profiles (Normalized Feature Means)", fontsize=14)
ax.legend()
plt.tight_layout()
plt.savefig("src/figures/cluster_profiles.png", dpi=150, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# cluster sizes
sizes = rfm["Cluster"].value_counts().sort_index()
axes[0].bar(
    [cluster_labels[c] for c in sizes.index], sizes.values,
    color=cluster_colors[:n_clusters], edgecolor="white"
)
for i, v in enumerate(sizes.values):
    axes[0].text(i, v + 20, str(v), ha="center",
                 fontsize=10, fontweight="bold")
axes[0].set_title("Customers per Cluster")
axes[0].set_ylabel("Count")

# average monetary
avg_monetary = rfm.groupby("Cluster")["Monetary"].mean().sort_index()
axes[1].bar(
    [cluster_labels[c] for c in avg_monetary.index], avg_monetary.values,
    color=cluster_colors[:n_clusters], edgecolor="white"
)
for i, v in enumerate(avg_monetary.values):
    axes[1].text(i, v + 20, f"${v:,.0f}", ha="center",
                 fontsize=10, fontweight="bold")
axes[1].set_title("Average Monetary Value per Cluster")
axes[1].set_ylabel("Avg Monetary ($)")

plt.tight_layout()
plt.savefig("src/figures/cluster_sizes_monetary.png",
            dpi=150, bbox_inches="tight")
plt.show()

print("\nCluster Summary")
summary = rfm.groupby("Cluster").agg(
    Count=("CustomerID", "count"),
    Avg_Recency=("Recency", "mean"),
    Avg_Frequency=("Frequency", "mean"),
    Avg_Monetary=("Monetary", "mean"),
).round(2)
summary.index = [cluster_labels[c] for c in summary.index]
print(summary.to_string())
