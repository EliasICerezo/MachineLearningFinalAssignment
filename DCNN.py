import keras
from sklearn.model_selection import train_test_split
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from keras.utils import to_categorical
import time
from keras.applications.resnet import ResNet50



def build_model(input_shape,num_classes):
  model = ResNet50(input_shape = input_shape, classes = num_classes,
                   weights = None)
  return model

# file_name = "stats/dcnn_mnist.csv"
file_name = "stats/dcnn_cifar10.csv"
try:
  df = pd.read_csv(file_name)
except FileNotFoundError as e:
  df = pd.DataFrame(columns=['execno', 'acc', 'loss', 'exec_time'])
  df.to_csv(file_name, index=False)

# Data preprocessing
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
# x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
ishape = x_train.shape[1:]

n_samples = x_train.shape[0]
n_samples_test = x_test.shape[0]
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
n_classes = 10
y_train = to_categorical(y_train, n_classes)
y_test = to_categorical(y_test, n_classes)

start = time.time()
model = build_model(ishape,n_classes)
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=1)
metrics = model.evaluate(x_test, y_test, batch_size = 128, verbose=1)
end = time.time()

df = pd.read_csv(file_name)
df = df.append({'execno': 0, 'acc': metrics[1], 'loss': metrics[0], 'exec_time': end-start},
          ignore_index=True)
df.to_csv(file_name, index=False)

