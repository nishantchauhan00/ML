from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
import numpy as np

classifier = Sequential()

# Convolution + Pooling Layer
classifier.add(
    Convolution2D(
        32,
        (3, 3),
        input_shape=(64, 64, 3),
        activation="relu"
    )
)
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())

classifier.add(Dense(128, activation="relu"))
classifier.add(Dense(1, activation="sigmoid"))

classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_data = train_datagen.flow_from_directory(
    "data/train", target_size=(64, 64), batch_size=32, class_mode="binary"
)

test_data = test_datagen.flow_from_directory(
    "data/test", target_size=(64, 64), batch_size=32, class_mode="binary"
)

classifier.fit_generator(
    train_data,
    samples_per_epoch=1227,
    nb_epoch=20,
    validation_data=test_data,
    nb_val_samples=360,
)
# val_loss: 0.3712 - val_accuracy: 0.8556