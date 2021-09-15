#KNN Algorithm euclidean distance

import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd
import sklearn

df = pd.read_csv('breast-cancer-wisconsin.data')

df.replace('?', -9999999, inplace=True)
df.drop(['id'],1,inplace=True)

X = np.array(df.drop(['class'],1))
Y = np.array(df['class'])

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, Y_train)

ACC = clf.score(X_test, Y_test)

print(ACC)

example_measures = np.array([[4,2,1,1,1,2,3,2,1], [4,2,1,1,1,1,2,1,2]])

example_measures = example_measures.reshape(len(example_measures), -1)

prediciton = clf.predict(example_measures)
print(prediciton)

