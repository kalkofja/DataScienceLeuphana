from tensorflow import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train/255.
x_test = x_test/255.

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

model = Sequential()

model.add(Conv2D(10, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(Flatten())
model.add(Dense(10, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train.reshape(-1, 28, 28, 1), y_train, epochs=20, batch_size=1000)

data = K.eval(model.layers[0].weights[0])
print(data[:, :, :, 0].reshape(3, 3))

plt.figure(figsize=(6, 6))
for i in range(0, 10):
    ax = plt.subplot(2, 5, i + 1)
    plt.imshow(data[:, :, :, i].reshape(3, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

model2 = Sequential()
model2.add(Conv2D(
    10,
    kernel_size=(3, 3),
    activation="sigmoid",
    input_shape=(28, 28, 1),
    weights=model.layers[0].get_weights()
))

result = model2.predict(x_test[0].reshape(-1, 28, 28, 1))
print(result.shape)

plt.figure(figsize=(10, 10))
for i in range(0, 10):
    ax = plt.subplot(2, 5, i + 1)
    plt.imshow(result[0][:, :, i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()




