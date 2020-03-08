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
from sklearn import linear_model
import numpy as np

full_df = pd.read_csv(r'C:\Users\exter\OneDrive\Documents\My Programs\Kaggle\Housing Price Competition'\
                      +r'\train.csv')
test_Xs = pd.read_csv(r'C:\Users\exter\OneDrive\Documents\My Programs\Kaggle\Housing Price Competition'\
                      +r'\test.csv')

usefulCols = ['LotArea', 'OverallQual', 'OverallCond']
# no bad data in the predictor and target
# full_df[['LotArea', 'SalePrice']].isnull().sum()
Y = full_df['SalePrice']

X = full_df[usefulCols]
#if single predictor
if X.shape[1]==1:
    X = np.array(X).reshape(-1, 1)
lm=linear_model.LinearRegression()
model=lm.fit(X, Y)

#get the test set
predictions=lm.predict(test_Xs[usefulCols])
print(predictions[0:5])

# write results to a submission file
test_Xs['SalePrice']=predictions
results = test_Xs[['Id', 'SalePrice']]
results.to_csv(r'C:\Users\exter\OneDrive\Documents\My Programs\Kaggle\Housing Price Competition'\
                      +r'\submission.csv', index=False)
