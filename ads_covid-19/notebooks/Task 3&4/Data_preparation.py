#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np

pd.set_option('display.max_rows',500)


# In[14]:


#!.[CRISP_DM].(../ads_covid-19/reports/figures/CRISP_DM.png)


# John hopkins https://github.com/CSSEGISandData/COVID-19

# GITHUB csv data
# 
# git clone/pull https://github.com/CSSEGISandData/COVID-19.git

# In[3]:


data_path='../ads_covid-19/data/raw/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
pd_raw=pd.read_csv(data_path)


# In[5]:


pd_raw.head()


# In[6]:


pd_raw.columns[4:]


# In[7]:


time_idx=pd_raw.columns[4:]


# In[8]:


df_plot=pd.DataFrame({
    'date':time_idx})
df_plot.head()


# In[10]:


pd_raw['Country/Region']


# In[12]:


pd_raw[pd_raw['Country/Region']=='US'].iloc[:,4::].sum(axis=0)


# In[13]:


country_list=['Italy',
              'US',
              'Spain',
              'Germany',
              'Korea,South',
             ]


# In[17]:


for each in country_list:
    df_plot[each]=np.array(pd_raw[pd_raw['Country/Region']==each].iloc[:,4::].sum(axis=0))


# In[21]:


get_ipython().run_line_magic('matplotlib', 'inline')
df_plot.set_index('date').plot()


# Data Type Date

# In[25]:


df_plot.head()


# In[36]:


time_idx=[datetime.strptime( each,"%m/%d/%y") for each in df_plot.date]
time_str=[each.strftime('%Y-%m-%d') for each in time_idx]


# In[35]:


from datetime import datetime


# In[39]:


df_plot['date']=time_idx
type(df_plot['date'][0])


# In[40]:


#datetime.strptime(df_plot.date[0],"%m/%d/%y")


# In[41]:


#time_idx=[datetime.strptime( each,"%m/%d/%y") for each in df_plot.date]


# In[42]:


#time_idx[0:5]


# In[43]:


#time_str=[each.strftime('%Y-%m-%d') for each in time_idx]
#time_str[0:5]


# In[44]:


df_plot.head()


# # Rational data model - defining a primary key

# In[45]:


data_path='../ads_covid-19/data/raw/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
pd_raw=pd.read_csv(data_path)
pd_raw.head()


# In[47]:


pd_data_base=pd_raw.rename(columns={'Country/Region':'country',
                                    'Province/State':'state'})


# In[48]:


pd_data_base=pd_data_base.drop(['Lat','Long'],axis=1)
pd_data_base.head()


# In[51]:


test_pd=pd_data_base.set_index(['state','country']).T


# In[52]:


test_pd.columns


# In[54]:


test_pd.stack(level=[0,1]).reset_index()


# # Group-by apply

# In[139]:


pd_JH_data=pd.read_csv('../ads_covid-19/data/processed/COVID_relational_confirmed.csv',sep=';',parse_dates=[0])
pd_JH_data=pd_JH_data.sort_values('date',ascending=True).reset_index(drop=True).copy()
pd_JH_data.head()


# In[140]:


test_data=pd_JH_data[((pd_JH_data['country']=='US')|
                     (pd_JH_data['country']=='Germany'))&
                    (pd_JH_data['date']>'2020-03-20')]


# In[141]:


test_data


# In[142]:


#test_data.groupby(['country']).agg(np.max)


# In[173]:


# %load ../ads_covid-19/src/features/build_features.py
import numpy as np
from sklearn import linear_model
reg=linear_model.LinearRegression(fit_intercept=True)

def get_doubling_time_via_regression(in_array):
    '''Use a linear regression to approximate the doubling rate'''
    
    y=np.array(in_array)
    x=np.arange(-1,2).reshape(-1,1)
    
    assert len(in_array)==3
    reg.fit(x,y)
    intercept=reg.intercept_
    slope=reg.coef_
    
    return intercept/slope

if __name__=='__main__':
    test_data=np.array([2,4,6])
    result=get_doubling_time_via_regression(test_data)
    print('the test slope is:'+str(result))

