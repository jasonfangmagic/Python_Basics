#KNN Algorithm euclidean distance
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from collections import Counter
style.use('fivethirtyeight')
import pandas as pd
import random

#k features
dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5], [7,7], [8,6]]}

new_features = [5,7]

# for i in dataset:
#     for ii in dataset[i]:
#         plt.scatter(ii[0], ii[1], s=100, color = i)

[[plt.scatter(ii[0], ii[1], s=100, color = i) for ii in dataset [i]] for i in dataset]
plt.scatter(new_features[0], new_features[1])
plt.show()

#use np sqrt to not limit 2D features
def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!!!')
    distance = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distance.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distance) [:k]]
    # print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k

    print(vote_result, confidence)

    return vote_result, confidence

result = k_nearest_neighbors(dataset, new_features, k=3)
print(result)

# [[plt.scatter(ii[0], ii[1], s=100, color = i) for ii in dataset [i]] for i in dataset]
# plt.scatter(new_features[0], new_features[1], s = 100, color = result)
# plt.show()

ACCS = []
for i in range(25):
    df = pd.read_csv('breast-cancer-wisconsin.data')
    df.replace('?', 999999, inplace = True)
    df.drop(['id'], 1, inplace = True)
    #conver value to float
    full_data = df.astype(float).values.tolist()
    print(full_data[:10])
    random.shuffle(full_data)
    print(20*'#')
    print(full_data[:5])

    test_size = 0.2
    train_set = {2:[], 4:[]}
    test_set = {2:[], 4:[]}
    train_data = full_data[:-int(test_size*len(full_data))]
    test_data = full_data[-int(test_size*len(full_data)):]

    for i in train_data:
        train_set[i[-1]].append(i[:-1])

    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    correct = 0
    total = 0

    for group in test_set:
        for data in test_set[group]:
            vote, confidence = k_nearest_neighbors(train_set, data, k=5)
            if group == vote:
                correct +=1
            # else:
            #     print(confidence)
            total +=1
    print('ACC: ', correct/total)
    ACCS.append(correct/total)

print(sum(ACCS)/len(ACCS))
 






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

