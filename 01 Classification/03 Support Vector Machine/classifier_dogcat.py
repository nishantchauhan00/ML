import cv2.cv2 as cv2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
import pickle

size = 100
features = 1500

dir_data = Path("data")
folders = [d for d in dir_data.iterdir() if d.is_dir()]
# cat-0 dog-1
img_data = []
class_data = []

for i, folder in enumerate(folders):
    for img in folder.iterdir():
        img = cv2.imread("data/" + folder.name + "/" + img.name)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_data.append(img.ravel())
        class_data.append(0 if folder.name == "cat" else 1)

pca = PCA(n_components=features)
img_data = pca.fit_transform(np.array(img_data))

X, X_test, Y, Y_test = train_test_split(
    img_data, np.array(class_data), test_size=0.2, shuffle=True
)

classifier = SVC(kernel="rbf")
classifier.fit(X, Y)

Y_pred = classifier.predict(X_test)
# around 65-70%

print("%.2f" % (accuracy_score(Y_test, Y_pred) * 100))
print(confusion_matrix(Y_test, Y_pred))

f = open("data/data_dump.pkl", "wb")
pickle.dump(classifier, f)
