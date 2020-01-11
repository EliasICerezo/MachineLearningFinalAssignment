import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from sklearn.metrics import accuracy_score, mean_squared_error
import pandas as pd
import random
import time

try:
  df = pd.read_csv("stats/svm.csv")
except FileNotFoundError as e:
  df = pd.DataFrame(columns=['execno', 'acc', 'mse', 'exec_time'])
  df.to_csv("stats/svm.csv")

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = x_train.shape[0]
n_samples_test = x_test.shape[0]
x_train = x_train.reshape((n_samples, -1))
x_test = x_test.reshape((n_samples_test, -1))
print(x_train.shape)
print(x_test.shape)

for i in range(30):
  df = pd.read_csv("stats/svm.csv")
  start = time.time()
  # Create a classifier: a support vector classifier
  classifier = svm.SVC(gamma=0.001, verbose=True, kernel='rbf',
                      random_state= random.randint(0,101))

  classifier.fit(x_train, y_train)
  predicted = classifier.predict(x_test)

  acc = accuracy_score(y_test, predicted)
  mse = mean_squared_error(y_test, predicted)
  print("Accuracy: "+str(acc))
  print("MSE: "+str(mse))
  end = time.time()
  df.append({'execno': i, 'acc': acc, 'mse': mse, 'exec_time': end-start},
            ignore_index=True)
  df.to_csv("stats/svm.csv")
