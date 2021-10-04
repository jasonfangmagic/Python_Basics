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

sns.set_style('whitegrid')
fig, axes = plt.subplots()
fig.set_size_inches(11.7,8.27) #size of A4 paper
h = plt.hist(list1,stacked=True, bins=30, rwidth=1, label=mylabels)
plt.title("Movie Budget Distribution", fontsize = 35,
          color = 'DarkBlue', fontname='Console')
plt.ylabel('Number of Movies', fontsize = 35,color = 'Red')
plt.xlabel('Budget', fontsize = 35, color = 'Green')
plt.xticks(fontsize= 20 )
plt.yticks(fontsize= 20)
plt.legend(prop={'size':20}, frameon = True,
           fancybox=True, shadow=True, framealpha=1)
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
g.set(xlim = (0,100), ylim = (0,100))
for ax in g.axes.flat:
    ax.plot((0,100), (0,100), c='gray', ls = '--')
g.add_legend()
plt.show()

g = sns.FacetGrid(data, row = 'Genre', col = 'Year', hue = 'Genre')
g = g.map(plt.hist, "Budget")
plt.show()

sns.set_style('dark', {'axes.facecolor':'black'})
f, axes = plt.subplots(2,2, figsize=(15,15))
v1 = sns.kdeplot(data.Budget, data.AudienceRatings,
                 shade=True,shade_lowest=True, cmap='inferno',ax = axes[0,0])
v1b = sns.kdeplot(data.Budget, data.AudienceRatings,
                 cmap='cool',ax = axes[0,0])

v2 = sns.kdeplot(data.Budget, data.Ratings,
                 shade=True,shade_lowest=True, cmap='inferno',ax = axes[0,1])
v2b = sns.kdeplot(data.Budget, data.Ratings,
                 cmap='cool',ax = axes[0,1])
v1.set(xlim=(-40,230))
v2.set(xlim=(-40,230))

v3 = sns.violinplot(data=data, x='Year', y = 'Budget',
                    palette='YlOrRd',ax = axes[1,0])

v4 = sns.kdeplot(data.Ratings, data.AudienceRatings,
                   shade = True, shade_lowest = False, cmap = 'Blues_r',ax = axes[1,1])
v4b = sns.kdeplot(data.Ratings, data.AudienceRatings,
                   cmap = 'gist_gray_r', ax = axes[1,1])
plt.show()


