# aclat

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('machine_learning/5_Association_Rule_Learning/Section 28 - Apriori/Python/Market_Basket_Optimisation.csv', header = None)

#number 7501
print(len(dataset))
#number 20
print(len(dataset.columns))

#have to be string so we use list
#recreate the dataframe
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])


# Training the Apriori model on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2,  max_length = 2)

# Visualising the results
results = list(rules)
print(results)
