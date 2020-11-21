'''
Q1 Data augumentation for the CNN for CIFAR-10 dataset
'''
import tensorflow.keras as keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# Hyper parameters
batch_size = 64
num_classes = 10
epochs = 5

# Load Cifar 10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', np.shape(x_train))
print(x_train.shape[0], 'train samples before preprocessing')
print(x_test.shape[0], 'test samples before preprocessing')

# One-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Feature normalization
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
datagen.fit(x_train)

# Keras Model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

#Please use Keras ImageDataGenerator to perfrom image data augumentation.
#Hint: You can use (https://keras.io/preprocessing/image/#imagedatagenerator)

# Save results
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=epochs, validation_data=(x_test, y_test), shuffle=True)
now = datetime.datetime.now().strftime("%H-%M-%S")
pd.DataFrame(history.history).to_csv('Results/history-' + now + '.csv')
data_df = pd.DataFrame(history.history).values

# Plot results
plt.plot(data_df[:, 0], data_df[:, 1], label='Training')
plt.plot(data_df[:, 0], data_df[:, 3], label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Q1 Epochs vs Training/Validation Accuracy')
plt.legend()
