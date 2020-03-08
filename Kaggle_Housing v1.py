# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 20:01:38 2020

Kaggle Competition
https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

v1. Using only a few of the numerical columns

RMSE=0.4

@author: exter
"""


import pandas as pd
import statsmodels.api as sm # import statsmodels 
import numpy as np
import matplotlib.pyplot as plt

full_df = pd.read_csv(r'C:\Users\exter\OneDrive\Documents\My Programs\Kaggle\Housing Price Competition'\
                      +r'\train.csv')
test_Xs = pd.read_csv(r'C:\Users\exter\OneDrive\Documents\My Programs\Kaggle\Housing Price Competition'\
                      +r'\test.csv')

#usefulCols = [ 'OverallQual', 'GrLivArea']

#usefulCols = ['LotArea']
usefulCols = ['GrLivArea']

# in isolation, GrLivArea gives a much higher R-Sq than LotArea


Y = full_df['SalePrice']

X = full_df[usefulCols]
#if single predictor
if X.shape[1]==1:
    X = np.array(X).reshape(-1, 1)

X = sm.add_constant(X)
# Note the difference in argument order
model = sm.OLS(Y, X).fit() ## sm.OLS(output, input)

print(model.summary())

pred_train = model.predict(X)
#get the test set
pred_Xs = sm.add_constant(test_Xs[usefulCols])
predictions=model.predict(pred_Xs)
#print(predictions[0:5])

# write results to a submission file
test_Xs['SalePrice']=predictions
results = test_Xs[['Id', 'SalePrice']]
results.to_csv(r'C:\Users\exter\OneDrive\Documents\My Programs\Kaggle\Housing Price Competition'\
                      +r'\submission.csv', index=False)

print("Submission.csv generated\n")


#______________________
# USEFUL DIAGNOSTICS
# _____________________

# visualize data
# full_df['OverallCond'].plot.hist()

# no bad data in the predictor and target
# full_df[['LotArea', 'SalePrice']].isnull().sum()

# pairwise correlation
# X.corr()

# pairwise plots
# import seaborn as sns
# sns.pairplot(X)

# scatter plots
# plt.scatter(np.log(full_df['LotArea']), Y)
