import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
matplotlib.use('svg')


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train/255
x_test = x_test/255

encoder_input = tf.keras.Input(shape=(28, 28, 1), name='img')
x = tf.keras.layers.Flatten()(encoder_input)
encoder_output = tf.keras.layers.Dense(100, activation='relu')(x)

encoder = tf.keras.Model(encoder_input, encoder_output, name='encoder')

decoder_input = tf.keras.layers.Dense(100,activation='relu')(encoder_output)
x = tf.keras.layers.Dense(784,activation='relu')(decoder_input)
decoder_output = tf.keras.layers.Reshape((28, 28, 1))(x)

opt = tf.keras.optimizers.Adam(learning_rate=0.001, decay = 1e-6)

autoencoder = tf.keras.Model(encoder_input, decoder_output, name='autoencoder')
autoencoder.summary()

autoencoder.compile(opt, loss='mse')
autoencoder.fit(x_train, x_train, epochs=3, batch_size=32, validation_split=0.1)
