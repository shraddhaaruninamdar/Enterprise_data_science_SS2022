#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
mpl.rcParams['figure.figsize']=(20,16)
pd.set_option('display.max_rows',500)

import plotly.graph_objects as go


# In[2]:


#!.[CRISP_DM].(../ads_covid-19/reports/figures/CRISP_DM.png)


# John hopkins https://github.com/CSSEGISandData/COVID-19

# GITHUB csv data
# 
# git clone/pull https://github.com/CSSEGISandData/COVID-19.git

# In[3]:


df_analyse=pd.read_csv('../ads_covid-19/data/processed/COVID_small_flat_table.csv',sep=';',
                       parse_dates=[0])
df_analyse.sort_values('date',ascending=True).tail()


# # Helper functions

# In[4]:


def quick_plot(x_in,df_input,y_scale='log',slider=False):
    fig=go.Figure()
    for each in df_input.columns:
        fig.add_trace(go.Scatter(
                        x=x_in,
                        y=df_input[each],
                        name=each,
                        opacity=0.8))
    fig.update_layout(autosize=True,
        width=1024,
        height=768,
        font=dict(
            family="PT Sans, monospace",
            size=18,
            color="#7f7f7f"
            )
        )
    fig.update_yaxes(type='log'), #range=[0.1,2])
    fig.update_xaxes(tickangle=-45,
                nticks=20,
                tickfont=dict(size=14,color="#7f7f7f")
                )
    if slider==True:
        fig.update_layout(xaxis_rangeslider_visible=True)
    fig.show()
    
    


# In[5]:


quick_plot(df_analyse.date,
            df_analyse.iloc[:,1:],
            y_scale='log',
            slider=True)


# In[6]:


threshold=100


# In[7]:


compare_list=[]
for pos,country in enumerate(df_analyse.columns[1:]):
    compare_list.append(np.array(df_analyse[country][df_analyse[country]>threshold]))


# In[8]:


pd_sync_timelines=pd.DataFrame(compare_list,index=df_analyse.columns[1:]).T


# In[9]:


pd_sync_timelines['date']=np.arange(pd_sync_timelines.shape[0])


# In[10]:


quick_plot(pd_sync_timelines.date,
          pd_sync_timelines.iloc[:,:-1],
          y_scale='log',
          slider=True)


# In[11]:


def doubling_rate(N_0,t,T_d):
    return N_0*np.power(2,t/T_d)


# In[12]:


max_days=60

norm_slopes={
    'doubling every day':doubling_rate(100,np.arange(max_days),1),
    'doubling every two days':doubling_rate(100,np.arange(max_days),2),
    'doubling every 4 day':doubling_rate(100,np.arange(max_days),3),
    'doubling every 10 days':doubling_rate(100,np.arange(max_days),4),
}


# In[13]:


pd_sync_timelines_w_slope=pd.concat([pd.DataFrame(norm_slopes),pd_sync_timelines],axis=1)


# In[52]:


quick_plot(pd_sync_timelines_w_slope.date,
          pd_sync_timelines_w_slope.iloc[:,0:4],
          y_scale='log',
          slider=True)


# # Understanding Linear Regression

# In[15]:


from sklearn import linear_model
reg=linear_model.LinearRegression(fit_intercept=False)


# In[16]:


l_vec=len(df_analyse['Germany'])
x=np.arange(l_vec-5).reshape(-1,1)
y=np.log(np.array(df_analyse['Germany'][5:]))


# In[17]:


reg.fit(x,y)


# In[18]:


x_hat=np.arange(l_vec).reshape(-1,1)
y_hat=reg.predict(x_hat)


# In[19]:


LR_inspect=df_analyse[['date','Germany']].copy()


# In[20]:


LR_inspect['prediction']=np.exp(y_hat)


# In[54]:


quick_plot(LR_inspect.date,
          LR_inspect.iloc[:,1:],
          y_scale='log',
          slider=True)


# # Doubling rate- Piecewise linear regression
# 
# ###### Doubling rate defines the no. of days required to double the infection rate

# In[22]:


from sklearn import linear_model
reg=linear_model.LinearRegression(fit_intercept=True)

#l_vec=len(df_analyse['Germany'])
#x=np.arange(l_vec-50).reshape(-1,1)
#y=np.array(df_analyse['Germany'][50:])


# In[23]:


from scipy import signal


# In[24]:


df_analyse=pd.read_csv('../ads_covid-19/data/processed/COVID_small_flat_table.csv',sep=';',
                       parse_dates=[0])  
country_list=df_analyse.columns[1:]


# In[25]:


#country_list=df_analyse.columns[2:]
for each in country_list:
    df_analyse[each+'_filter']=signal.savgol_filter(df_analyse[each],
                                               5,
                                               1)


# In[26]:


filter_cols=['Italy_filter','US_filter','Spain_filter','Germany_filter']


# In[27]:


start_pos=5
quick_plot(df_analyse.date[start_pos:],
          df_analyse[filter_cols].iloc[start_pos:,:],
          y_scale='log',
          slider=True)


# In[28]:


df_analyse.head()


# In[29]:


reg.fit(x,y)


# In[30]:


reg.intercept_


# In[31]:


reg.coef_


# In[32]:


reg.coef_/reg.intercept_


# In[33]:


def get_doubling_time_via_regression(in_array):
    ''' Use a linear regression to approximate the doubling rate'''
    
    y = np.array(in_array)
    X = np.arange(-1,2).reshape(-1, 1)
    
    assert len(in_array)==3
    reg.fit(X,y)
    intercept=reg.intercept_
    slope=reg.coef_
    
    return intercept/slope


# In[34]:


def doubling_time(in_array):
    ''' Use a classical doubling time formular, 
     see https://en.wikipedia.org/wiki/Doubling_time '''
    y = np.array(in_array)
    return len(y)*np.log(2)/np.log(y[-1]/y[0])


# In[35]:


# calculate slope of regression of last x days
# use always a limited number of days to approximate the triangle, attention exponential base assumption
days_back = 3 # this gives a smoothing effect
for pos,country in enumerate(country_list):
    df_analyse[country+'_DR']=df_analyse[country].rolling(
                                window=days_back,
                                min_periods=days_back).apply(get_doubling_time_via_regression, raw=False)


# In[36]:


# run on all filtered data
days_back = 3 # this gives a smoothing effect
for pos,country in enumerate(filter_cols):
    df_analyse[country+'_DR']=df_analyse[country].rolling(
                                window=days_back,
                                min_periods=days_back).apply(get_doubling_time_via_regression, raw=False)


# In[37]:


# cross check the matematical 
df_analyse['Germany_DR_math']=df_analyse['Germany'].rolling(
                                window=days_back,
                                min_periods=days_back).apply(doubling_time, raw=False)


# In[38]:


# run on all filtered data
days_back = 3 # this gives a smoothing effect
for pos,country in enumerate(filter_cols):
    df_analyse[country+'_DR']=df_analyse[country].rolling(
                                window=days_back,
                                min_periods=days_back).apply(get_doubling_time_via_regression, raw=False)


# In[39]:


df_analyse.columns


# In[58]:


start_pos=40
quick_plot(df_analyse.date[start_pos:],
           df_analyse.iloc[start_pos:,[11,12,14]], #
           y_scale='linear',
           slider=True)


# In[61]:


start_pos=40
quick_plot(df_analyse.date[start_pos:],
           df_analyse.iloc[start_pos:,[11,16]], #17,18,19   # US comparison 12,17
           y_scale='linear',
           slider=True)


# In[ ]:





# In[ ]:




