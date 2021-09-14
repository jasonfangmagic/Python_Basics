import math

import pandas as pd
import quandl
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.utils import shuffle
import sklearn
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import pickle


df = quandl.get('WIKI/GOOGL')

print(df.head())

df  = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]

df['HL_PCT']= (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close']
df['PCT_change']= (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']


df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecasst = 'Adj. Close'

df.fillna(-9999999, inplace = True)

forecasst_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecasst].shift(-forecasst_out)


print(df.tail())

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X = X[:-forecasst_out]
X_lately = X[-forecasst_out:]

# J = [2,3,4,5,7,8,9,10,11,12]
# forecasst_J= int(math.ceil(0.01*len(J)))
# print(forecasst_out)
# J1 = J[:-forecasst_J]
# print(J1)
# J2 = J[-forecasst_J:]
# print(J2)

df.dropna(inplace = True)
Y = np.array(df['label'])

print(len(X), len(Y))

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

clf = LinearRegression()
clf.fit(X_train, Y_train)

ACC = clf.score(X_test, Y_test)

print(ACC)

#use svm
clf = svm.SVR(kernel='poly')
clf.fit(X_train, Y_train)


ACC = clf.score(X_test, Y_test)
print(ACC)

forecasst_set = clf.predict(X_lately)

print(forecasst_set, ACC)


df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecasst_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix +=one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc = 4)
plt.xlabel ('Date')
plt.ylabel ('Price')
plt.show()


#algorithm

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

xs = np.array([1,2,3,4,5,6], dtype=np.float64)
ys = np.array([5,4,2,3,5,7], dtype=np.float64)
plt.scatter(xs,ys)
plt.show()

def best_fit_slope(xs, ys):
    m =( ((mean(xs)*mean(ys)) - mean(xs*ys)) /
          ((mean(xs)*mean(xs)) - mean(xs*xs)))

    b = mean(ys) - m*mean(xs)
    return m, b

m,b = best_fit_slope(xs, ys)



print(m, b)

regression_line = [(m*x)+b for x in xs]


#the same
# for x in xs:
#     regression_line.append((m*x)+b)

#same
plt.scatter(xs, ys)
plt.plot(xs, regression_line)
plt.show()

def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig)**2)

def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

