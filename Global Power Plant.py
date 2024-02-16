#!/usr/bin/env python
# coding: utf-8

# # Global Power Plant Database
# Importing necessary Libraries

# In[197]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import scipy as stats
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[198]:


# Reading the csv file from dataset
df = pd.read_csv("database_IND.csv")
df


# In[ ]:





# In[199]:


df.head()


# In[200]:


df.isnull().sum()


# In[201]:


df.columns


# In[202]:


df.describe()


# In[203]:


df.drop(["other_fuel1", "other_fuel2", "other_fuel3", "owner", "wepp_id","generation_data_source", "estimated_generation_gwh"], axis=1, inplace=True)


# In[204]:


df.shape


# In[205]:


df.nunique()


# In[206]:


df.drop(['country','country_long','year_of_capacity_data','generation_gwh_2013','name','gppd_idnr','url','generation_gwh_2019'], axis=1, inplace=True)


# In[207]:


df.shape


# In[208]:


df


# In[209]:


df['geolocation_source'].fillna(df['geolocation_source'].mode()[0], inplace=True)


# In[210]:


df.fillna(df.median()[0], inplace=True)


# In[211]:


df.isnull().sum()


# # Exploratory Data Analysis(EDA)

# In[212]:


sns.countplot(x ='primary_fuel',data = df)
plt.show()


# In[213]:


sns.scatterplot(x='generation_gwh_2014',y='capacity_mw', data = df)


# In[214]:


sns.scatterplot(x='generation_gwh_2015',y='capacity_mw', data = df)


# In[215]:


sns.scatterplot(x='generation_gwh_2016',y='capacity_mw', data = df)


# In[216]:


sns.scatterplot(x='generation_gwh_2017',y='capacity_mw', data = df)


# In[217]:


sns.scatterplot(x='generation_gwh_2018',y='capacity_mw', data = df)


# In[218]:


sns.scatterplot(x='commissioning_year',y='capacity_mw', data = df)


# In[219]:


sns.scatterplot(x='latitude',y='capacity_mw',data=df)


# In[220]:


sns.scatterplot(x='commissioning_year',y='primary_fuel',data=df)


# In[221]:


sns.scatterplot(x='generation_gwh_2018',y='primary_fuel',data=df)


# In[222]:


plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),cmap='Purples',annot=True)


# In[223]:


df.info()


# In[224]:


df['primary_fuel'].unique()


# In[225]:


df['geolocation_source'].unique()


# In[226]:


from sklearn.preprocessing import LabelEncoder


# In[227]:


encoder = LabelEncoder()
df['primary_fuel'] = encoder.fit_transform(df['primary_fuel'])
df['geolocation_source'] = encoder.fit_transform(df['geolocation_source'])
df['source'] = encoder.fit_transform(df['source'])


# In[228]:


df['geolocation_source'].unique()


# In[229]:


df.info()


# # Z-Score

# In[230]:


from scipy.stats import zscore
z=np.abs(zscore(df))
z


# In[231]:


threshold=3
print(np.where(z>3))


# In[232]:


df_new=df[(z<3).all(axis=1)]


# In[233]:


df.shape


# In[234]:


df_new.shape


# In[235]:


x=df.drop(['capacity_mw'],axis=1)
y=df['capacity_mw']


# Splitting data

# In[236]:


from sklearn.model_selection import train_test_split


# In[237]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.27,random_state=95)


# In[238]:


x_train.shape


# In[239]:


x_test.shape


# # Model Making

# In[240]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import r2_score


# In[241]:


#Linear Regression


# In[242]:


from sklearn.linear_model import LinearRegression


LR=LinearRegression()
LR.fit(x_train,y_train)
print(LR.score(x_train,y_train))
LR_predict=LR.predict(x_test)


# In[243]:


print(mean_squared_error(LR_predict,y_test))
print(mean_absolute_error(LR_predict,y_test))
print(r2_score(LR_predict,y_test))


# In[244]:


#Ridge Regression


# In[245]:


from sklearn.linear_model import Ridge

R=Ridge()
R.fit(x_train,y_train)
print(R.score(x_train,y_train))
R_predict=R.predict(x_test)


# In[246]:


print(mean_squared_error(LR_predict,y_test))
print(mean_absolute_error(LR_predict,y_test))
print(r2_score(LR_predict,y_test))


# In[247]:


#Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor

GBR=GradientBoostingRegressor()
GBR.fit(x_train,y_train)
print(GBR.score(x_train,y_train))
GBR_PRED=GBR.predict(x_test)


# In[248]:


print(mean_squared_error(LR_predict,y_test))
print(mean_absolute_error(LR_predict,y_test))
print(r2_score(LR_predict,y_test))


# In[249]:


#Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

RF=RandomForestRegressor()
RF.fit(x_train,y_train)
print(RF.score(x_train,y_train))
RF_PRED=RF.predict(x_test)


# In[250]:


print(mean_squared_error(LR_predict,y_test))
print(mean_absolute_error(LR_predict,y_test))
print(r2_score(LR_predict,y_test))


# In[251]:


#Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor

DTR=DecisionTreeRegressor()
DTR.fit(x_train,y_train)
print(DTR.score(x_train,y_train))
DTR_PRED=DTR.predict(x_test)


# In[252]:


print(mean_squared_error(LR_predict,y_test))
print(mean_absolute_error(LR_predict,y_test))
print(r2_score(LR_predict,y_test))


# # Cross Validation

# In[253]:


#Linear regression

from sklearn.model_selection import cross_val_score

print(cross_val_score(LR,x,y,cv=5).mean())


# In[254]:


# Random Forest Regressor

from sklearn.model_selection import cross_val_score

print(cross_val_score(RF,x,y,cv=5).mean())


# In[255]:


#Decision Tree Regressor

from sklearn.model_selection import cross_val_score

print(cross_val_score(DTR,x,y,cv=5).mean())


# In[265]:


# Ridge regression

from sklearn.model_selection import cross_val_score

print(cross_val_score(R,x,y,cv=5).mean())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




