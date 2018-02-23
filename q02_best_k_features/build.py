# %load q02_best_k_features/build.py
# Default imports

'''import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression, SelectKBest

X = data.iloc[:,:-1]
y = data.iloc[:,-1]
#print X.iloc[0]
selector = SelectKBest(f_regression,k=20).fit(X,y)
#print X_new.get_params(deep=True)
X_new = selector.transform(X)
scores = selector.pvalues_
#print scores
#print scores
# Write your solution here:
cols_sel = selector.get_support(indices=True)
print X.columns[cols_sel]
'''


# Default imports

import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression


# Write your solution here:

def percentile_k_features(dataframe, k=20):
    X = dataframe.iloc[:, :-1]
    y = dataframe.iloc[:, -1]
    model = f_regression

    skb = SelectPercentile(model, k)
    predictors = skb.fit_transform(X, y)
    scores = list(skb.scores_)
    #print scores
    top_k_index = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:predictors.shape[1]]
    #print top_k_index
    top_k_predictors = [X.columns[i] for i in top_k_index]

    return top_k_predictors

#print percentile_k_features(data)


