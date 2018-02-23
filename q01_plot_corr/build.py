# %load q01_plot_corr/build.py
# Default imports
import pandas as pd
from matplotlib.pyplot import yticks, xticks, subplots, set_cmap, imshow
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

def plot_corr(data,size=11):    
    num_columns =data.select_dtypes(include=['float64','int64'])
    plt.figure(figsize=(10,6))
    sns.heatmap(num_columns.corr(),cmap='YlOrRd')
    #plt.xticks(range(18),num_columns)
    #plt.yticks(range(18),num_columns)
    return plt.show()

#plot_corr(data,11)

