from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train/255.
x_test = x_test/255.

encoder = Sequential()
encoder.add(Conv2D(5, kernel_size=(3, 3), padding="same", input_shape=(28, 28, 1), activation="relu"))
encoder.add(MaxPooling2D(pool_size=(2, 2)))
encoder.add(Conv2D(10, kernel_size=(3, 3), padding="same", activation="relu"))
encoder.add(MaxPooling2D(pool_size=(2, 2)))


decoder = Sequential()
decoder.add(Conv2D(10, kernel_size=(3, 3), padding="same", input_shape=(7, 7, 10), activation="relu"))
decoder.add(UpSampling2D(size=(2, 2)))
decoder.add(Conv2D(5, kernel_size=(3, 3), padding="same", activation="relu"))
decoder.add(UpSampling2D(size=(2, 2)))
decoder.add(Conv2D(1, kernel_size=(3, 3), padding="same", activation="sigmoid"))


autoencoder = Sequential()
autoencoder.add(encoder)
autoencoder.add(decoder)

autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.summary()

autoencoder.fit(x_train.reshape(-1, 28, 28, 1), x_train.reshape(-1, 28, 28, 1), epochs=10, batch_size=100)

print(encoder.layers)

model2 = Sequential()
model2.add(Conv2D(
    5,
    kernel_size=(3, 3),
    activation="sigmoid",
    input_shape=(28, 28, 1),
    weights=encoder.layers[0].get_weights()
))

result = model2.predict(x_test[0].reshape(-1, 28, 28, 1))

plt.figure(figsize=(6, 6))
for i in range(0, 5):
    ax = plt.subplot(2, 3, i + 1)
    plt.imshow(result[0][:, :, i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

# image = x_train[0]
# image_predicted = autoencoder.predict(image.reshape(-1, 28, 28, 1))
# print(image_predicted.shape)
#
#
# encoded = encoder.predict(x_test[0].reshape(-1, 28, 28, 1))

# plt.figure(figsize=(6, 6))
# plt.subplot(3, 3, 1)
# plt.imshow(encoded[0, :, :, 0])
# plt.gray()
#
# plt.subplot(3, 3, 2)
# plt.imshow(encoded[0, :, :, 1])
# plt.gray()
#
# plt.subplot(3, 3, 3)
# plt.imshow(encoded[0, :, :, 2])
# # plt.gray()
#
# plt.subplot(3, 3, 4)
# plt.imshow(encoded[0, :, :, 3])
# plt.gray()
#
# plt.subplot(3, 3, 5)
# plt.imshow(encoded[0, :, :, 4])
# plt.gray()
#
# plt.show()
#
# decoded = decoder.predict(encoded)
# plt.imshow(decoded.reshape(28, 28))
# plt.show()
