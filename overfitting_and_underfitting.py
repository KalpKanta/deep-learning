import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
(x_train, y_train), (x_test,y_test) = mnist.load_data()
(x_train, x_test) = (x_train / 255, x_test / 255)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
x_train = x_train.reshape(x_train.shape[0], 28,28,1)
x_test = x_test.reshape(x_test.shape[0], 28,28,1)

idg = ImageDataGenerator(rotation_range = 40, width_shift_range = 0.1, height_shift_range = 0.1, zoom_range = 0.1)
idg.fit(x_train)
#hyperparameters
initial_learning_rate = 0.01
epoch = 20 #epoch is used for the freuency the model will go throught the entire dataset
batch_size = 64 #number of samples proccessed before model updates
def lr_schedule(epoch, lr):
    if epoch > 0 and epoch%3 == 0:
        return lr * 0.5

    return lr

model= Sequential([
    Flatten(input_shape = (28,28)), #flatten - converts 2d images to 1d images
    Dense(500, activation = "relu"),
    Dropout(0.2), 
    Dense(64, activation = "relu"),
    Dropout(0.2),
    Dense(10, activation = "softmax")
]) 

optimizer = SGD(learning_rate = initial_learning_rate)
model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])

es = EarlyStopping(monitor = "val_loss", patience = 3, restore_best_weights = True)
lr = LearningRateScheduler(lr_schedule)
history = model.fit(idg.flow(x_train, y_train, batch_size = batch_size), epochs = epoch, validation_data = (x_test, y_test), callbacks = [es, lr])
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"\ntest accuracy = {test_accuracy * 100:.2f}%")
plt.plot(history.history["accuracy"], label = "train accuracy")
plt.plot(history.history["val_accuracy"], label = "validation space accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.show()