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

import matplotlib.pyplot as plt
import seaborn as sns

#joint plot
sns.set()
j = sns.jointplot(data=data, x= 'Ratings',y = 'AudienceRatings', kind='hex' )
#hex plot you can see clusters
plt.show()

m1 = sns.distplot(data.AudienceRatings , bins=15)
plt.show()

m2 = sns.distplot(data.Ratings , bins=15)
plt.show()

sns.set_style('white')
n1 = plt.hist(data.AudienceRatings, bins=15)
plt.show()

#stacked histgram
#Filter
data[data.Genre == 'Drama'].Budget
h1 = plt.hist(data[data.Genre == 'Action'].Budget)
h1 = plt.hist(data[data.Genre == 'Drama'].Budget)
h1 = plt.hist(data[data.Genre == 'Thriller'].Budget)
plt.show()

#lsit
list = [data[data.Genre == 'Action'].Budget,
        data[data.Genre == 'Drama'].Budget,
        data[data.Genre == 'Thriller'].Budget
        ]
h2 = plt.hist(list, stacked=True)
plt.show()

data.Genre.cat.categories

for gen in data.Genre.cat.categories:
    print(gen)

def myplot(data, gen = data.Genre.cat.categories):
    for i in data.Genre.cat.categories:
        plt.hist(data[data.Genre == i].Budget,stacked=True)
        plt.show()

myplot (data, gen= "Action")

#stacked
list1 = []
mylabels = []
for gen in data.Genre.cat.categories:
        list1.append(data[data.Genre == gen].Budget)
        mylabels.append(gen)
print(list1)

h = plt.hist(list1,stacked=True, bins=30, rwidth=1, label=mylabels)
plt.legend()
plt.show()

#kernel density estimation plot
vis1 = sns.lmplot(data= data, x='Ratings', y = "AudienceRatings",
                  fit_reg= False, hue = 'Genre', size = 7, aspect = 1)
plt.show()

sns.set_style('darkgrid')
vis2 = sns.kdeplot(data.Ratings, data.AudienceRatings,
                   shade = True, shade_lowest = False, cmap = 'Reds'
                  )
#combine two graphs to make it clearer
vis3 = sns.kdeplot(data.Ratings, data.AudienceRatings,
                   cmap = 'Reds'
                  )
plt.show()

vis4 = sns.kdeplot(data.Budget, data.AudienceRatings
                   )
plt.show()

#create subplots
sns.set_style('dark')
f, axes = plt.subplots(1,2, figsize=(12,6), sharex=True, sharey=True)
vis4 = sns.kdeplot(data.Budget, data.AudienceRatings, ax = axes[0])
vis5 = sns.kdeplot(data.Budget, data.Ratings, ax = axes[1])
vis4.set(xlim=(-40,220))
plt.show()

#violin plot
sns.set_style('dark')
f, axes = plt.subplots(1,2, figsize=(12,6), sharex=True, sharey=True)
vis6 = sns.violinplot(data=data, x='Genre', y = 'Ratings', ax = axes[0])
vis7 = sns.boxplot(data=data, x='Genre', y = 'Ratings', ax = axes[1])
plt.show()


vis8 = sns.violinplot(data=data[data.Genre=='Drama'], x='Year', y = 'Ratings')
plt.show()

#create facet grid

g = sns.FacetGrid(data, row = 'Genre', col = 'Year', hue = 'Genre')
kws = dict(s=50, linewidth=0.5, edgecolor='black')
g = g.map(plt.scatter, "Ratings", "AudienceRatings",**kws)
plt.show()

g = sns.FacetGrid(data, row = 'Genre', col = 'Year', hue = 'Genre')
g = g.map(plt.hist, "Budget")
plt.show()