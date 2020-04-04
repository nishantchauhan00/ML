import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import collections
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("data_mall.csv")
datac = data.copy()

# plt.bar(["M", "F"], collections.Counter(data["Gender"]).values())
# plt.hist(data["Spending Score (1-100)"], bins=50)
# plt.show()

encoder = LabelEncoder()
data["Gender"] = encoder.fit_transform(data["Gender"])
data.drop("CustomerID", axis=1, inplace=True)

sc = StandardScaler()
data.iloc[:, 1:] = sc.fit_transform(data.iloc[:, 1:])

# inertia = []

# for i in range(1, 11):
#     cluster = KMeans(n_clusters=i)
#     cluster.fit(data.iloc[:, :])
#     inertia.append(cluster.inertia_)

# plt.plot(np.arange(1, 11, 1), inertia)
# plt.plot(np.arange(1, 11, 1), inertia,'*', c="r")
# plt.xlabel("Number of Clusters")
# # Inertia = sum of distance of centroid from each points
# plt.ylabel("Inertia")
# plt.title("Selecting no. of clusters")
# plt.show()

#
# Main Clustering
#
cluster_centers = []
cluster_labels = []
cluster = KMeans(n_clusters=4)
cluster.fit(data.iloc[:, :])

cluster_centers.append(sc.inverse_transform(cluster.cluster_centers_[:, 1:]))
cluster_labels.append(cluster.labels_)

# 2d
a = plt.figure(1)
plt.scatter(
    datac["Spending Score (1-100)"], datac["Annual Income (k$)"], c=cluster_labels[0]
)
plt.xlabel("Spending Score (1-100)")
plt.ylabel("Annual Income (k$)")
plt.plot(
    cluster_centers[0][:, 2], cluster_centers[0][:, 1], "o", markersize=10, c="black"
)

# 3d
b = plt.figure(2)
ax = Axes3D(plt.gcf())
ax.scatter(
    xs=datac["Spending Score (1-100)"],
    ys=datac["Annual Income (k$)"],
    zs=datac["Age"],
    c=cluster_labels[0],
)
ax.scatter(
    xs=cluster_centers[0][:, 2],
    ys=cluster_centers[0][:, 1],
    zs=cluster_centers[0][:, 0],
    marker="*",
    c="black",
)
ax.set_xlabel("Spending Score (1-100)")
ax.set_ylabel("Annual Income (k$)")
ax.set_zlabel("Age")

plt.show()

