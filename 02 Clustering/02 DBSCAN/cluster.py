# Density Based Spatial Clustering of Applications with Noise
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, scale
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

data = pd.read_csv("data_mall.csv")

encoder = LabelEncoder()
data["Gender"] = encoder.fit_transform(data["Gender"])

# data.iloc[:, 2:] = scale(data.iloc[:, 2:])
# scaler = StandardScaler()
# data.iloc[:, 2:] = scaler.fit_transform(data.iloc[:, 2:])
data.iloc[:, 2] = (data.iloc[:, 2] - np.min(data.iloc[:, 2])) / (
    np.max(data.iloc[:, 2]) - np.min(data.iloc[:, 2])
)
data.iloc[:, 3] = (data.iloc[:, 3] - np.min(data.iloc[:, 3])) / (
    np.max(data.iloc[:, 3]) - np.min(data.iloc[:, 3])
)
data.iloc[:, 4] = (data.iloc[:, 4] - np.min(data.iloc[:, 4])) / (
    np.max(data.iloc[:, 4]) - np.min(data.iloc[:, 4])
)

cluster = DBSCAN(eps=0.15)
cluster_labels = cluster.fit_predict(data.iloc[:, 1:])

n_clusters = max(cluster_labels) + 1
print("Number of Clusters: ", n_clusters)

plt.figure(1)
plt.scatter(
    data["Spending Score (1-100)"], data["Annual Income (k$)"], c=cluster_labels
)
plt.xlabel("Spending Score (1-100)")
plt.ylabel("Annual Income (k$)")
for i in range(0, len(cluster_labels)):
    if cluster_labels[i] == -1:
        plt.scatter(
            data["Spending Score (1-100)"][i], data["Annual Income (k$)"][i], c="black"
        )

plt.figure(2)
ax = Axes3D(plt.gcf())
ax.scatter(
    xs=data["Spending Score (1-100)"],
    ys=data["Age"],
    zs=data["Annual Income (k$)"],
    c=cluster_labels,
)
ax.set_xlabel("Spending Score (1-100)")
ax.set_ylabel("Age")
ax.set_zlabel("Annual Income (k$)")
for i in range(0, len(cluster_labels)):
    if cluster_labels[i] == -1:
        ax.scatter(
            xs=data["Spending Score (1-100)"][i],
            ys=data["Age"][i],
            zs=data["Annual Income (k$)"][i],
            c="black",
        )

plt.show()

