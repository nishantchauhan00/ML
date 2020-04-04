import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

data = pd.read_csv("data.csv")

data.drop("User ID", axis=1, inplace=True)
data["EstimatedSalary"] = [int(n / 1000) for n in data["EstimatedSalary"]]

encoder = LabelEncoder()
data["Gender"] = encoder.fit_transform(data["Gender"])

X, X_test, Y, Y_test = train_test_split(
    data.iloc[:, :3], data["Purchased"], test_size=0.2
)

classifier = GaussianNB()
classifier.fit(X, Y)

Y_pred = classifier.predict(X_test)
print(accuracy_score(Y_test, Y_pred))
print(confusion_matrix(Y_test, Y_pred))

a = plt.figure(1)
plt.scatter(X_test["Age"], X_test["EstimatedSalary"], c=Y_test, marker="*")
b = plt.figure(2)
plt.scatter(X_test["Age"], X_test["EstimatedSalary"], c=Y_pred)
plt.show()
