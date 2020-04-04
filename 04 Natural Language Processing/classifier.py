import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# stopwords
stopwords_1 = set(stopwords.words("english"))
stopwords_2 = set((pd.read_csv("data\\stopwords_en.csv", header=None))[0])
stopwords = stopwords_1.union(stopwords_2)

data = pd.read_csv("data\\data_irony.csv")

# cleaning data
ps = PorterStemmer()
corpus = []
for i in range(0, data.shape[0]):
    text = data["comment_text"][i]
    text = re.sub(
        r"(https?:\/\/)(\s)*(www\.)?(\s)*((\w|\s)+\.)*([\w\-\s]+\/)*([\w\-]+)((\?)?[\w\s]*=\s*[\w\%&]*)*",
        " ",
        text,
    )
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if not word in stopwords]
    text = " ".join(text)
    corpus.append(text)

# creating bag of words - creates array of words as column and text as row
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
Y = data["label"]

# classifier
X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)
classifier = GaussianNB()
classifier.fit(X, Y)

Y_pred = classifier.predict(X_test)

print(confusion_matrix(Y_test, Y_pred), "\n")
print(accuracy_score(Y_test, Y_pred) * 100)

