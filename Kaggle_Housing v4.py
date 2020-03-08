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
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import statsmodels.api as sm # import statsmodels 
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.api import abline_plot

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

df_train = pd.read_csv(r'C:\Users\exter\OneDrive\Documents\My Programs\Kaggle\Housing Price Competition'\
                      +r'\train.csv')
df_test = pd.read_csv(r'C:\Users\exter\OneDrive\Documents\My Programs\Kaggle\Housing Price Competition'\
                      +r'\test.csv')
# example of one hot encoding for a neural network

seed = 7

# load the dataset
def load_dataset(df):
	# retrieve numpy array
	dataset = df.values
	# split into input (X) and output (y) variables
	X = dataset[:, :-1]
	y = dataset[:,-1]
	# format all fields as string
	X = X.astype(str)
	# reshape target to be a 2d array
	y = y.reshape((len(y), 1))
	return X, y

# prepare input data
def prepare_inputs(X_train, X_test):
	ohe = OneHotEncoder()
	ohe.fit(X_train)
	X_train_enc = ohe.transform(X_train)
	X_test_enc = ohe.transform(X_test)
	return X_train_enc, X_test_enc

# prepare target
def prepare_targets(y_train, y_test):
	le = LabelEncoder()
	le.fit(y_train)
	y_train_enc = le.transform(y_train)
	y_test_enc = le.transform(y_test)
	return y_train_enc, y_test_enc

# load the dataset
# X, y = load_dataset(df_train)

X = df_train.drop('SalePrice', axis=1)
y = df_train['SalePrice']
X_test = df_test
X_combined = pd.concat([X, X_test])

# split into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)
# prepare input data
# X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)

# We create the preprocessing pipelines for both numeric and categorical data.
numeric_features = ['GrLivArea', 'OverallQual']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['GarageType', 'MSSubClass', 'SaleType']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

X_train_processed = preprocessor.fit(df_train).transform(df_train)

X_train_processed = preprocessor.fit(X_combined).transform(X_combined)
#X2.toarray() 
# breakpoint()

# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(33, input_dim=33, kernel_initializer='normal', activation='relu'))
	model.add(Dense(15, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

model = KerasRegressor(build_fn=baseline_model, epochs=5, batch_size=5, verbose=0)
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                      ('mlp', model)
                      ])
    
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(pipeline, X, y, cv=kfold)
print("Standardized: %f%% (%f%%)" % (results.mean()*100, results.std()*100))

X_test = np.ones(33)

model.fit(X_train_processed, y)

X_test_processed = preprocessor.fit(df_test).transform(df_test)
prediction = model.predict(X_test_processed)

plt.scatter(y, prediction)
#accuracy_score(Y_test, prediction)

#potential improvements
# add log functionTransformer to Y and some Xs

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
