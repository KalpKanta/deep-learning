import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import SGD #Stochastic Gradient Descent
from tensorflow.keras.callbacks import LearningRateScheduler #This is to adjust learning rate during training

(x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, x_test) = (x_train / 255, x_test / 255) 
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
initial_learning_rate = 0.01
epoch = 20 #epoch is used for the freuency the model will go throught the entire dataset
batch_size = 64 #number of samples proccessed before model updates
def lr_schedule(epoch, lr):
    if epoch > 0 and epoch%3 == 0:
        return lr * 0.5

    return lr

model= Sequential([
    Flatten(input_shape = (28,28)),
    Dense(128, activation = "relu"),
    Dense(64, activation = "relu"),
    Dense(10, activation = "softmax")
])

optimizer = SGD(learning_rate = initial_learning_rate)
model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])
lr_scheduler = LearningRateScheduler(lr_schedule)
history = model.fit(x_train, y_train, batch_size = batch_size, epochs = epoch, validation_data = (x_test,y_test), callbacks = [lr_scheduler])
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(test_accuracy * 100)