import tensorflow
# import keras
from tensorflow import keras

import sklearn

from sklearn import linear_model
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1","G2", "G3", "studytime", "failures", "absences"]]

print(data.head())

predict = "G3"

X = np.array(data.drop([predict], 1))

Y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)


#save the best model
# best = 0

# for _ in range(30):
#
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
#
#     linear = linear_model.LinearRegression()
#
#     linear.fit(x_train, y_train)
#
#     acc = linear.score(x_test, y_test)
#
#     print(acc)
#
#     if acc > best:
#         best = acc
#     #save the model
#         with open("student model.pickle", "wb") as f:
#             pickle.dump(linear, f)

pickle_in = open("student model.pickle", "rb")
linear = pickle.load(pickle_in)

print("Co: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])


linear.fit(x_train, y_train)

acc = linear.score(x_test, y_test)

print(acc)

#plotting

p = "G1"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
