# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 20:01:38 2020

Kaggle Competition
https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

formulas in the model:
    https://www.statsmodels.org/stable/examples/notebooks/generated/formulas.html
    
v3. Using GLM

@author: exter
"""


import pandas as pd
import statsmodels.api as sm # import statsmodels 
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.api import abline_plot

full_df = pd.read_csv(r'C:\Users\exter\OneDrive\Documents\My Programs\Kaggle\Housing Price Competition'\
                      +r'\train.csv')
test_Xs = pd.read_csv(r'C:\Users\exter\OneDrive\Documents\My Programs\Kaggle\Housing Price Competition'\
                      +r'\test.csv')

logTransform = True
formula='np.log(SalePrice) ~ GrLivArea + OverallQual + YearBuilt + YearRemodAdd +TotalBsmtSF'

usefulCols = ['GrLivArea']

# in isolation, GrLivArea gives a much higher R-Sq than LotArea

#X = sm.add_constant(X)
model = ols(formula=formula, data=full_df).fit()

print(model.summary())

pred_train = model.predict(full_df)
#get the test set
predictions=model.predict(test_Xs)
if logTransform:
    predictions = np.exp(predictions)
#print(predictions[0:5])



# write results to a submission file
test_Xs['SalePrice']=predictions
results = test_Xs[['Id', 'SalePrice']]
results.to_csv(r'C:\Users\exter\OneDrive\Documents\My Programs\Kaggle\Housing Price Competition'\
                      +r'\submission.csv', index=False)

print("\nSubmission.csv OVERWRITTEN.\n")


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
