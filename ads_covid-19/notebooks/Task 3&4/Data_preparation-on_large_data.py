#!/usr/bin/env python
# coding: utf-8

# In[167]:


import pandas as pd
import numpy as np

pd.set_option('display.max_rows',500)


# In[168]:


#!.[CRISP_DM].(../ads_covid-19/reports/figures/CRISP_DM.png)


# John hopkins https://github.com/CSSEGISandData/COVID-19

# GITHUB csv data
# 
# git clone/pull https://github.com/CSSEGISandData/COVID-19.git

# # Group-by apply

# In[170]:


pd_JH_data=pd.read_csv('../ads_covid-19/data/processed/covid_relational_confirmed.csv',sep=';',parse_dates=[0])
pd_JH_data=pd_JH_data.sort_values('date',ascending=True).reset_index(drop=True).copy()
pd_JH_data.tail()


# In[171]:


test_data=pd_JH_data[((pd_JH_data['country']=='US')|
                     (pd_JH_data['country']=='Germany'))
                    ]


# In[172]:


test_data.head()


# In[173]:


test_data.groupby(['country']).agg(np.max)


# In[174]:


#load ../ads_covid-19/src/features/build_features_shraddha.py
import numpy as np
from sklearn import linear_model
reg = linear_model.LinearRegression(fit_intercept=True)

def get_doubling_time_via_regression(in_array):
    ''' Use a linear regression to approximate the doubling rate'''

    y = np.array(in_array)
    X = np.arange(-1,2).reshape(-1, 1)

    assert len(in_array)==3
    reg.fit(X,y)
    intercept=reg.intercept_
    slope=reg.coef_

    return intercept/slope


# In[175]:


test_data.groupby(['state','country']).agg(np.max)


# In[176]:


#test_data.groupby(['state','country']).apply(get_doubling_time_via_regression)


# In[177]:


def rolling_reg(df_input,col='confirmed'):
    days_back=3
    result=df_input[col].rolling(
                window=days_back,
                min_periods=days_back).apply(get_doubling_time_via_regression,raw=False)
    return result


# In[178]:


test_data[['state','country','confirmed']].groupby(['state','country']).apply(rolling_reg,'confirmed')


# In[179]:


test_data


# In[180]:


pd_DR_result=pd_JH_data[['state','country','confirmed']].groupby(['state','country']).apply(rolling_reg,'confirmed').reset_index()


# In[181]:


pd_DR_result=pd_DR_result.rename(columns={'confirmed':'confirmed_DR',
                             'level_2':'index'})
pd_DR_result.head()


# In[182]:


pd_JH_data=pd_JH_data.reset_index()
pd_JH_data.head()


# In[183]:


pd_result_larg=pd.merge(pd_JH_data,pd_DR_result[['index','confirmed_DR']],on=['index'],how='left')
pd_result_larg


# # Filtering the data with groupby apply 

# In[184]:


from scipy import signal

def savgol_filter(df_input,column='confirmed',window=5):
    ''' Savgol Filter which can be used in groupby apply function 
        it ensures that the data structure is kept'''
    window=5, 
    degree=1
    df_result=df_input
    
    filter_in=df_input[column].fillna(0) # attention with the neutral element here
    
    result=signal.savgol_filter(np.array(filter_in),
                           5, # window size used for filtering
                           1)
    df_result[column+'_filtered']=result
    return df_result
        


# In[185]:


pd_filtered_result=pd_JH_data[['state','country','confirmed']].groupby(['state','country']).apply(savgol_filter).reset_index()


# In[186]:


pd_result_larg=pd.merge(pd_result_larg,pd_filtered_result[['index','confirmed_filtered']],on=['index'],how='left')
pd_result_larg.head()


# # Filtered doubling rate

# In[187]:


pd_filtered_doubling=pd_result_larg[['state','country','confirmed_filtered']].groupby(['state','country']).apply(rolling_reg,'confirmed_filtered').reset_index()

pd_filtered_doubling=pd_filtered_doubling.rename(columns={'confirmed_filtered':'confirmed_filtered_DR',
                             'level_2':'index'})

pd_filtered_doubling.tail()


# In[188]:


pd_result_larg=pd.merge(pd_result_larg,pd_filtered_doubling[['index','confirmed_filtered_DR']],on=['index'],how='left')
pd_result_larg.tail()


# In[189]:


mask=pd_result_larg['confirmed']>100
pd_result_larg['confirmed_filtered_DR']=pd_result_larg['confirmed_filtered_DR'].where(mask, other=np.NaN) 


# In[190]:


pd_result_larg[pd_result_larg['country']=='Germany'].tail()


# In[191]:


pd_result_larg.to_csv('../ads_covid-19/data/processed/COVID_final_set.csv',sep=';',index=False)


# In[ ]:





# In[ ]:




