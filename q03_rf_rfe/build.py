# %load q03_rf_rfe/build.py
# Default imports
'''import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE

data = pd.read_csv('data/house_prices_multivariate.csv')

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

clf = RandomForestClassifier(random_state=0)
clf.fit(X, y)
for i in range(0,17):
    print sorted(zip(clf.feature_importances_, X))[17:36][i][1]
#print list(zip(X, clf.feature_importances_))
#clf.predict_proba(X)
#clf.scores_
# Your solution code here'''

import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


# Your solution code here
def rf_rfe(dataframe):
    X = dataframe.iloc[:, :-1]
    y = dataframe.iloc[:, -1]
    model = RandomForestClassifier()

    rfe = RFE(model, round(len(X.columns) / 2, 0)).fit(X, y)
    top_features = []
    #print rfe.ranking_
    for i in range(len(rfe.ranking_)):
        if rfe.ranking_[i] == 1:
            top_features.append(X.columns[i])

    return top_features

#print rf_rfe(data)




