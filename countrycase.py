import pandas as pd
import numpy as np

data = pd.read_csv("Demographic.csv")
print(data.head())

len(data)
data.columns
len(data.columns)
data.head()
data.tail()
data.info()
data.describe().transpose()
data.columns = ['CountryName', 'CountryCode', 'BirthRate', 'InternetUsers',
       'IncomeGroup']
data.columns

data[['CountryName','BirthRate']]

data.CountryName

data[4:6][['CountryName','BirthRate']]

#data['myCal'] = data.BirthRate*data.InternetUsers
#data = data.drop('myCal',1)

Filter = data.InternetUsers < 2
Filter.head()
data[Filter]

Filter2 = data.BirthRate > 40
Filter2.head()

Filter & Filter2

data[data.IncomeGroup == 'High income']

data.IncomeGroup.unique()

#locate individual elements
#third row, fourth column
data.iat[3,4]
data.at[3,'BirthRate']

import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = 8,5

sns.set()
vis1 = sns.displot(data['InternetUsers'], bins=30)
plt.show()

vis2 = sns.boxplot(data = data, x= 'IncomeGroup', y= 'BirthRate')
plt.show()

#seaborn gallery
# https://seaborn.pydata.org/examples/index.html

#lmplot is linear regression model
vis3 = sns.lmplot(data = data, x='InternetUsers', y='BirthRate',
                  fit_reg=False, hue = 'IncomeGroup', size = 8,
                  scatter_kws = {"s":100})
plt.show()