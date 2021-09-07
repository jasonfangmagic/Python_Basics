import tensorflow
import keras
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import pandas as pd
import numpy as np

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1","G2", "G3", "studytime", "failures", "absences"]]

print(data.head())

predict = "G3"

X = np.array(data.drop([predict], 1))

Y = np.array(data[predict])

x_train, y_train, x_test, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)



