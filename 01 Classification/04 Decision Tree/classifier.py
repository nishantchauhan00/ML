import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv(
    "data_banknote.csv", names=["Variance", "Skewness", "Curtosis", "Entropy", "Class"]
)

# Not much difference
# sc = StandardScaler()
# data.iloc[:, :-1] = sc.fit_transform(data.iloc[:, :-1])

X, X_test, Y, Y_test = train_test_split(
    data.iloc[:, :-1], data.iloc[:, -1], test_size=0.2, shuffle=True
)

classifier = DecisionTreeClassifier(criterion="entropy")
classifier.fit(X, Y)

Y_pred = classifier.predict(X_test)

print("%.2f" % (accuracy_score(Y_test, Y_pred) * 100))
print(confusion_matrix(Y_test, Y_pred))

