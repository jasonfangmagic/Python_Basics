import pandas as pd
import os

data = pd.read_csv("Ratings.csv")
print(data.head())

data.columns

data.columns = ['Film', 'Genre', 'Ratings', 'AudienceRatings',
       'Budget', 'Year']

data.info()
data.Film = data.Film.astype("category")
data.Genre = data.Genre.astype("category")
data.Year = data.Year.astype("category")

data.Genre.unique
data.Genre.cat.categories