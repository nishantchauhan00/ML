import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

data = pd.read_csv('data\\data_fake_currency.csv')
X, X_test, Y, Y_test = train_test_split(data.iloc[:, :4], data.iloc[:, 4], test_size=0.2, shuffle=True)

classifier = SVC(kernel='linear')
classifier.fit(X, Y)

Y_pred = classifier.predict(X_test)

# print(classification_report(Y_test, Y_pred))
print(confusion_matrix(Y_test, Y_pred))
print("%.2f"%(accuracy_score(Y_test, Y_pred)*100))

