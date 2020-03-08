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
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import sklearn.feature_selection
import numpy as np
import matplotlib.pyplot as plt

#supress futurewrning

# import warnings filter
from warnings import simplefilter

logTransform = False
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

df_train = pd.read_csv(r'C:\Users\exter\OneDrive\Documents\My Programs\Kaggle\Housing Price Competition'\
                      +r'\train.csv')
df_test = pd.read_csv(r'C:\Users\exter\OneDrive\Documents\My Programs\Kaggle\Housing Price Competition'\
                      +r'\test.csv')
# example of one hot encoding for a neural network

seed = 7
# load the dataset
# X, y = load_dataset(df_train)

X = df_train.drop('SalePrice', axis=1)
y = df_train['SalePrice']
if logTransform:
    y = np.log(y)
X_test = df_test
X_combined = pd.concat([X, X_test])
numInTrain = X.shape[0]

# split into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)
# prepare input data
# X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)

# We create the preprocessing pipelines for both numeric and categorical data.
numeric_features = ['GrLivArea', 'OverallQual', 'MiscVal', 'OverallCond',  'TotRmsAbvGrd']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['MSSubClass', 'MSZoning','SaleType', 'LotShape', 'LandContour', 'Utilities', 'LandSlope', 
                        'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'YrSold', 'SaleCondition',
                        'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 
                        'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'Heating', 'HeatingQC', 'CentralAir', 
                        'Electrical', 'KitchenQual', 'Functional', 'GarageType', 'PavedDrive', 'PoolQC', 'YearRemodAdd'                
                               ]

categorical_features = ['MSSubClass', 'MSZoning','SaleType', 'LotShape', 'LandSlope', 
                        'Neighborhood', 'SaleCondition',
                        'Exterior1st', 'MasVnrType', 'ExterQual', 'ExterCond', 
                        'BsmtQual', 'BsmtCond',  'Heating', 'HeatingQC', 'CentralAir', 
                        'Electrical', 'KitchenQual', 'Functional', 'GarageType', 'PavedDrive', 'PoolQC', 'YearRemodAdd'                
                               ]

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

'''
X_combined_proc = preprocessor.fit(X_combined).transform(X_combined)

X_train_proc = X_combined_proc[:numInTrain, :]
X_test_proc = X_combined_proc[numInTrain:, :]
'''

''' PREPRCOESS: NUMERIC
'''
X_combined_num = X_combined[numeric_features]
X_combined_num = numeric_transformer.fit(X_combined_num).transform(X_combined_num)
X_combined_num = pd.DataFrame(data=X_combined_num, columns=numeric_features)

''' PERPROCESS: CATEGORICAL
'''
X_combined_cat = X_combined[categorical_features]
simpImp = SimpleImputer(strategy='constant', fill_value='missing')
simpImp.fit(X_combined_cat)
X_combined_cat = pd.DataFrame(data=simpImp.transform(X_combined_cat ), columns=X_combined_cat.columns)
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(X_combined_cat)
enc.categories_
catCols = enc.get_feature_names(categorical_features)
X_combined_cat = pd.DataFrame(data=enc.transform(X_combined_cat).toarray(), columns=catCols)

# Combine numeric + categorical
X_combined_proc = pd.concat([X_combined_num, X_combined_cat], axis=1)
X_train_proc = X_combined_proc[:numInTrain]
X_test_proc = X_combined_proc[numInTrain:]

# see what the best features are, using univariate criterion from regression
numTopFeatures=100
select = sklearn.feature_selection.SelectKBest(k=numTopFeatures)
selected_features = select.fit(X_train_proc,y)
indices_selected = selected_features.get_support(indices=True)
colnames_selected = list(X_train_proc.columns[i] for i  in indices_selected)

X_train_proc = X_train_proc[colnames_selected]
X_test_proc = X_test_proc[colnames_selected]


# define base model
def baseline_model(optimizer='rmsprop', init='glorot_uniform'):
	# create model
    model = Sequential()
    model.add(Dense(100, input_dim=100, kernel_initializer='normal', activation='relu'))
    model.add(Dense(50, kernel_initializer='normal', activation='relu'))
    model.add(Dense(15, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

model = KerasRegressor(build_fn=baseline_model, epochs=150, batch_size=5, verbose=0)
pipeline = Pipeline(steps=[
                        #('preprocessor', preprocessor), #preprossed already with the combined X data
                      ('mlp', model)
                      ])
    
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)


results = cross_val_score(pipeline, X_train_proc, y, cv=kfold)
print("Standardized: %f (%f)" % (np.log(-results).mean(), np.log(-results).std()*100))

''' Use the model to predict for X_test
'''
model.fit(X_train_proc, y)
prediction = model.predict(X_train_proc)
prediction_test = model.predict(X_test_proc)
if logTransform:
    prediction = np.exp(prediction)
    prediction_test = np.exp(prediction_test)
    plt.scatter(np.exp(y), prediction)
else:
    plt.scatter(y, prediction)

breakpoint()

# grid search epochs, batch size and optimizer
'''
# add CV
# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)
'''
optimizers = ['rmsprop', 'adam']
init = ['glorot_uniform', 'normal', 'uniform']
epochs = np.array([50, 100, 150])
batches = np.array([5, 10, 20])

param_grid = dict(optimizer=optimizers, nb_epoch=epochs, batch_size=batches, init=init)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X_train_proc, y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    mean_n = np.log(-mean)
    print("%f (%f) with: %r" % (mean_n, stdev, param))



#accuracy_score(Y_test, prediction)

#potential improvements
# add log functionTransformer to Y and some Xs

#simplify some of the variables:
# year sold: pre and post crisis
# avoid hardcoding input_dim
# https://stackoverflow.com/questions/47944463/specify-input-argument-with-kerasregressor
                
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
