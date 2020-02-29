#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastai.tabular import *


# In[2]:


pwd


# In[3]:


df = pd.read_csv('train.csv')


# In[4]:


df.shape


# In[5]:


df.columns


# In[6]:


df['SalePrice']


# In[7]:


cat_names = ['MSSubClass', 'MSZoning','Street','Alley','LotShape','LandContour','Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType','HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType','ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFinSF2', 'Heating','Fireplaces', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual','GarageCond', 'PavedDrive',  'PoolQC', 'Fence', 'MiscFeature', 'SaleType',
       'SaleCondition']

# cont_names = ['LotFrontage', 'SalePrice']
cont_names = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',        'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',        'GarageYrBlt','GarageCars', 'GarageArea','WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea','MiscVal', 'MoSold', 'YrSold']


# In[8]:


procs = [FillMissing, Categorify, Normalize]


# In[9]:


dep_var = 'SalePrice'


# In[10]:


get_ipython().run_line_magic('pinfo', 'TabularList.from_df')


# In[11]:


pwd


# In[31]:


path="D:\\python\\projects"


# In[53]:


testDf = pd.read_csv('test.csv')
testDf = testDf.drop('Id',axis=1)


# In[54]:


test = TabularList.from_df(testDf, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)


# In[55]:


data = (TabularList.from_df(df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
                           .split_by_rand_pct(valid_pct = 0.2)
                           .label_from_df(cols=dep_var)
                           .add_test(test)
                           .databunch())


# In[56]:


data.show_batch(rows=10)


# In[57]:


learn = tabular_learner(data, layers=[200,100], metrics=mean_squared_error)


# In[58]:


learn.fit(1, 1e-2)


# In[59]:


learn.lr_find()


# In[60]:


learn.recorder.plot()


# In[61]:


learn.unfreeze()


# In[62]:


learn.fit_one_cycle(8, max_lr=slice(1))


# In[63]:


preds, _ = learn.get_preds(ds_type=DatasetType.Test)


# In[64]:


preds


# In[65]:


result = preds.numpy()[:, 0]
result


# In[66]:


ids = [i for i in range(1461,2920)]


# In[67]:


d = {'Id': ids, 'SalePrice':result}
df2 = DataFrame(data=d)


# In[68]:


df2


# In[69]:


df2.to_csv('out.csv', index=False)


# In[ ]:




