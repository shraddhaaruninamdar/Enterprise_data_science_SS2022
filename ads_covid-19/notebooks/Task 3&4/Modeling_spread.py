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


# In[6]:


#!.[CRISP_DM].(../ads_covid-19/reports/figures/CRISP_DM.png)


# John hopkins https://github.com/CSSEGISandData/COVID-19

# GITHUB csv data
# 
# git clone/pull https://github.com/CSSEGISandData/COVID-19.git

# In[2]:


df_analyse=pd.read_csv('../Applied data science/COVID_small_flat_table.csv',sep=';',
                       parse_dates=[0])
df_analyse.sort_values('date',ascending=True).tail()


# # Helper functions

# In[6]:


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
    
    


# In[8]:


quick_plot(df_analyse.date,
            df_analyse.iloc[:,1:],
            y_scale='linear',
            slider=True)


# In[9]:


threshold=100


# In[12]:


compare_list=[]
for pos,country in enumerate(df_analyse.columns[1:]):
    compare_list.append(np.array(df_analyse[country][df_analyse[country]>threshold]))


# In[16]:


pd_sync_timelines=pd.DataFrame(compare_list,index=df_analyse.columns[1:]).T


# In[19]:


pd_sync_timelines['date']=np.arange(pd_sync_timelines.shape[0])


# In[22]:


quick_plot(pd_sync_timelines.date,
          pd_sync_timelines.iloc[:,:-1],
          y_scale='log',
          slider=True)


# In[23]:


def doubling_rate(N_0,t,T_d):
    return N_0*np.power(2,t/T_d)


# In[30]:


max_days=60

norm_slopes={
    'doubling every day':doubling_rate(100,np.arange(max_days),1),
    'doubling every two days':doubling_rate(100,np.arange(max_days),2),
    'doubling every 4 day':doubling_rate(100,np.arange(max_days),3),
    'doubling every 10 days':doubling_rate(100,np.arange(max_days),4),
}


# In[31]:


pd_sync_timelines_w_slope=pd.concat([pd.DataFrame(norm_slopes),pd_sync_timelines],axis=1)


# In[32]:


quick_plot(pd_sync_timelines_w_slope.date,
          pd_sync_timelines_w_slope.iloc[:,0:5],
          y_scale='log',
          slider=True)


# # Understanding Linear Regression

# In[60]:


from sklearn import linear_model
reg=linear_model.LinearRegression(fit_intercept=False)


# In[61]:


l_vec=len(df_analyse['Germany'])
x=np.arange(l_vec-5).reshape(-1,1)
y=np.log(np.array(df_analyse['Germany'][5:]))


# In[62]:


reg.fit(x,y)


# In[63]:


x_hat=np.arange(l_vec).reshape(-1,1)
y_hat=reg.predict(x_hat)


# In[64]:


LR_inspect=df_analyse[['date','Germany']].copy()


# In[65]:


LR_inspect['prediction']=np.exp(y_hat)


# In[68]:


quick_plot(LR_inspect.date,
          LR_inspect.iloc[:,1:],
          y_scale='log',
          slider=True)


# # Doubling rate- Piecewise linear regression

# In[113]:


from sklearn import linear_model
reg=linear_model.LinearRegression(fit_intercept=True)

#l_vec=len(df_analyse['Germany'])
#x=np.arange(l_vec-50).reshape(-1,1)
#y=np.array(df_analyse['Germany'][50:])


# In[114]:


from scipy import signal


# In[116]:


for each in country_list:
    df_analyse[each+'_filter']=signal.savgol_filter(df_analyse['US'],
                                               3,
                                               1)


# In[117]:


filter_cols=['US_filter','Spain_filter','Germany_filter','Korea, South_filter']


# In[118]:


start_pos=5
quick_plot(df_analyse.date[start_pos:],
          df_analyse[filter_cols].iloc[start_pos:,:],
          y_scale='log',
          slider=True)


# In[73]:


#reg.fit(x,y)


# In[74]:


#reg.intercept_


# In[75]:


#reg.coef_


# In[76]:


#reg.coef_/reg.intercept_


# In[78]:


def get_rate_via_regression(in_array):
    y=np.array(in_array)
    x=np.arange(-1,2).reshape(-1,1)
    
    assert len(in_array)==3
    
    reg.fit(x,y)
    intercept=reg.intercept_
    slope=reg.coef_
    
    return intercept/slope


# In[96]:


country_list=df_analyse.columns[1:]
for each in country_list:
    df_analyse[each+'_DR']=df_analyse[each].rolling(window=3,
                             min_periods=3).apply(get_rate_via_regression)


# In[97]:


quick_plot(df_analyse.date,df_analyse.iloc[40:,[6,7,8,9,10]],y_scale='linear')


# In[85]:


def doubling_time(in_array):
    y=np.array(in_array)
    return len(y)*np.log(2)/np.log(y[-1]/y[0])


# In[92]:


df_analyse['Germany_DT_wiki']=df_analyse['Germany'].rolling(window=3,
                                min_periods=3).apply(doubling_time)


# In[93]:


quick_plot(df_analyse.date,df_analyse.iloc[40:,[6,7]],y_scale='linear')


# In[ ]:




