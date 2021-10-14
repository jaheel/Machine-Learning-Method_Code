import numpy as np

from Model import Model
from Layers import Dense, Conv, Flatten, Pool

# from tensorflow.examples.tutorials.mnist import input_data

# mnist = input_data.read_data_sets("F:/Dataset/mnist/data/", one_hot = True)
# print(mnist.train.num_examples)
# print(mnist.test.num_examples)

import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.reshape(x_train[:10], (10, 28, 28, 1))
y_train = y_train[:10]
x_test = np.reshape(x_test[:10], (10, 28, 28, 1))
y_test = y_test[:10]
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

model = Model()

model.add(Conv(filters = 16, kernel_size = (7, 7), input_shape = (28, 28, 1), strides = (2, 2), padding = "VALID", activation ='tanh', name = 'C1'))
model.add(Conv(filters = 32, kernel_size = (5, 5), input_shape = (11, 11, 16), strides = (2, 2), padding = 'VALID', activation = 'tanh', name = 'C2'))
model.add(Flatten())
model.add(Dense(shape = (512, 64), activation = 'tanh', name = 'Dense'))
model.add(Dense(shape = (64, 10), activation = 'softmax', name = 'Dense2'))

model.set_loss_function('cross_entropy')
model.train_eval(X = x_train, y = y_train, learn_rate = 0.005, epochs = 20, X_test = x_test, y_test = y_test)

result = model.predict(x_test)
n = 0

for index, y in enumerate(y_test):
    pred = np.argmax(result[index])
    y_ = np.argmax(y)

    if pred == y_:
        n += 1

print("acc : ", n/len(y_test))