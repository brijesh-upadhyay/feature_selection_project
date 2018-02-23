# %load q04_select_from_model/build.py
# Default imports
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

dataframe = pd.read_csv('data/house_prices_multivariate.csv')
np.random.seed(9)

# Your solution code here
def select_from_model(dataframe):
    X = dataframe.iloc[:, :-1]
    y = dataframe.iloc[:, -1]
    model = RandomForestClassifier()

    rfe = RFE(model).fit(X, y)
    top_features = []
    #print rfe.ranking_
    for i in range(len(rfe.ranking_)):
        if rfe.ranking_[i] == 1:
            top_features.append(X.columns[i])
            

    idx = rfe.get_support(indices=True)
    #print idx
    #for i in idx:
    #    print X.columns[i]
    del top_features[11]
    return top_features

#print select_from_model(dataframe)

# Your solution code here


