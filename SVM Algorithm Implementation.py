import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

data_dict = {-1: np.array([[1, 7],
                           [2, 8],
                           [3, 8], ]),
             1: np.array([[5, 1],
                          [6, -1],
                          [7, 3], ])}

class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1:'r',-1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)

        # train
    def fit(self, data):
        pass

    def predict(self, features):
        # sign( x.w+b )
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)

        return classification


















import KNN
import numpy as np
from sklearn import preprocessing, model_selection, svm, neighbors
import pandas as pd
import sklearn

df = pd.read_csv('breast-cancer-wisconsin.data')

df.replace('?', -9999999, inplace=True)
df.drop(['id'],1,inplace=True)

X = np.array(df.drop(['class'],1))
Y = np.array(df['class'])

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

clf = svm.SVC(kernel='linear')
clf.fit(X_train, Y_train)

ACC = clf.score(X_test, Y_test)

print(ACC)


