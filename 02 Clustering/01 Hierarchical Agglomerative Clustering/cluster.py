import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv("data_mall.csv")

encoder = LabelEncoder()

data.iloc[:, 1] = encoder.fit_transform(data.iloc[:, 1])

# n_clusters=4, linkage="ward"
cluster = AgglomerativeClustering()
cluster = AgglomerativeClustering(n_clusters=4, linkage="ward")
cluster.fit(data.iloc[:, 1:])

plt.figure(1)
plt.scatter(
    data["Annual Income (k$)"],
    data["Spending Score (1-100)"],
    c=cluster.labels_,
    marker="*",
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")

plt.figure(2)
plt.scatter(data["Spending Score (1-100)"], data["Age"], c=cluster.labels_, marker="*")
plt.xlabel("Spending Score (1-100)")
plt.ylabel("Age")

plt.figure(3)
ax = Axes3D(plt.gcf())
ax.scatter(xs=data["Spending Score (1-100)"], ys=data["Annual Income (k$)"], zs=data["Age"], c=cluster.labels_)
ax.set_xlabel("Spending Score (1-100)")
ax.set_ylabel("Annual Income (k$)")
ax.set_zlabel("Age")

plt.show()
