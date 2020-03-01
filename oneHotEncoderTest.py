# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 12:54:13 2020

@author: exter
"""

from sklearn.preprocessing import OneHotEncoder
import pandas as pd

df_enc = pd.read_csv(r'C:\Users\exter\OneDrive\Documents\My Programs\Kaggle\Housing Price Competition'\
                      +r'\train.csv')

cols = ['MSSubClass', 'Street']
df_enc = df_enc[cols]

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(df_enc)
enc.categories_

new = enc.transform(df_enc).toarray()
colNames = enc.get_feature_names(cols)
new.shape
colNames.shape

'''
Example # 2
'''
enc = OneHotEncoder(handle_unknown='ignore')
X = [['Male', 1], ['Female', 3], ['Female', 2]]
enc.categories_

enc.transform([['Female', 1], ['Male', 4]]).toarray()


enc.inverse_transform([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]])


enc.get_feature_names(['gender', 'group'])


drop_enc = OneHotEncoder(drop='first').fit(X)
drop_enc.categories_

drop_enc.transform([['Female', 1], ['Male', 2]]).toarray()