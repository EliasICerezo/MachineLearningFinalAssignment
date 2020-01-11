import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from sklearn.metrics import accuracy_score, mean_squared_error

(x_train, y_train), (x_test, y_test) = mnist.load_data()


# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = x_train.shape[0]
n_samples_test = x_test.shape[0]
x_train = x_train.reshape((n_samples, -1))
x_test = x_test.reshape((n_samples_test, -1))
print(x_train.shape)
print(x_test.shape)

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001, verbose=True, kernel='rbf')

# We learn the digits on the first half of the digits
classifier.fit(x_train, y_train)

# Now predict the value of the digit on the second half:
predicted = classifier.predict(x_test)

acc = accuracy_score(y_test, predicted)
mse = mean_squared_error(y_test, predicted)

print("Accuracy: "+str(acc))
print("MSE: "+str(mse))