import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense  # densely connected layer
from keras.layers import Dropout
from keras.regularizers import l2
from sklearn.metrics import confusion_matrix, accuracy_score
from digits import digits

# training image - 4500, testing image - 500
X, Y, X_test, Y_test, encoder = digits()

classifier = Sequential()

# from keras.optimizers import SGD
# sgd = SGD(learning_rate=0.5, momentum=test_it)

# Input layer of 400 + Hidden layer of 30
classifier.add(
    Dense(40, activation="relu", kernel_initializer="he_normal", input_shape=(400,))
)
# OR classifier.add(Dense(30, activation="relu",input_dim=400))

# not much effect
# classifier.add(Dropout(0.2))

# Output layer of 10
classifier.add(Dense(10, kernel_initializer="he_normal", activation="softmax"))
# categorical/binary_crossentropy = logloss = cross entropy
classifier.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)

classifier.fit(np.array(X), np.array(Y), batch_size=10, epochs=30)

Y_pred = classifier.predict(np.array(X_test))

Y_test = encoder.inverse_transform(Y_test)
Y_pred = encoder.inverse_transform(Y_pred)

print("\nAccuracy Score: ", accuracy_score(Y_test, Y_pred))
print(confusion_matrix(Y_test, Y_pred))
