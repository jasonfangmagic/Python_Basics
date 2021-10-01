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
