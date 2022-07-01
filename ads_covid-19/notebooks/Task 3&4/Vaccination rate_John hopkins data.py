#!/usr/bin/env python
# coding: utf-8

# In[62]:


import subprocess
import os
from datetime import datetime

import requests
from bs4 import BeautifulSoup

import numpy as np

import json

import pandas as pd
pd.set_option('display.max_rows',500)

import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt


# In[63]:


mpl.rcParams['figure.figsize']=(16,9)
pd.set_option('display.max_rows',500)
sns.set(style='darkgrid')


# In[76]:


data_path='../ads_covid-19/data/raw/COVID-19/time_series_covid19_vaccine_doses_admin_global.csv'
pd_raw=pd.read_csv(data_path)


# In[77]:


pd_raw.head()


# # GITHUB csv data
# git clone/pull https://github.com/CSSEGISandData/COVID-19.git

# In[78]:


git_pull = subprocess.Popen( "/usr/bin/git pull" , 
                     cwd = os.path.dirname( '../ads_covid-19/data/raw/COVID-19/time_series_covid19_vaccine_doses_admin_global.csv'), 
                     shell = True, 
                     stdout = subprocess.PIPE, 
                     stderr = subprocess.PIPE )
(out, error) = git_pull.communicate()


print("Error : " + str(error)) 
print("out : " + str(out))


# In[79]:


time_idx=pd_raw.columns[12:]


# In[80]:


df_plot=pd.DataFrame({
    'date':time_idx})
df_plot.head()


# In[81]:


pd_raw['Country_Region']


# In[82]:


pd_raw[pd_raw['Country_Region']=='US'].iloc[:,12::].sum(axis=0)


# In[83]:


country_list=['Italy',
              'US',
              'Spain',
              'Germany',
              'Canada',
             ]


# In[84]:


for each in country_list:
    df_plot[each]=np.array(pd_raw[pd_raw['Country_Region']==each].iloc[:,12::].sum(axis=0))


# In[85]:


get_ipython().run_line_magic('matplotlib', 'inline')
df_plot.set_index('date').plot()


# # Data Type Date

# In[86]:


type(df_plot.date[0])


# In[87]:


df_plot.head()


# In[88]:


df_plot.to_csv('../ads_covid-19/data/processed/time_series_covid19_vaccine_small_flat_table',index=False)


# In[89]:


data_path='../ads_covid-19/data/processed/time_series_covid19_vaccine_small_flat_table'
pd_raw=pd.read_csv(data_path)
pd_raw


# In[90]:


data_path='../ads_covid-19/data/raw/COVID-19/time_series_covid19_vaccine_doses_admin_global.csv'
pd_raw=pd.read_csv(data_path)
pd_raw.head()


# In[91]:


pd_data_base= pd_raw.rename(columns={'Country_Region':'country',
                                     'Province_State':'state'})


# In[92]:


pd_data_base=pd_data_base.drop(['Long_','Combined_Key'],axis=1)
pd_data_base.head()


# In[93]:


test_pd=pd_data_base.set_index(['country','Population']).T


# In[94]:


test_pd


# In[95]:


#df['New_Row'] = df.Population.div(2022-06-11)


# In[96]:


test_pd.stack(level=[0,1]).reset_index()


# In[97]:


test_pd.rename(columns = {'level_0':'date', '':'Vaccination'}, inplace = True)


# In[98]:


pd_relational_model=pd_data_base.set_index(['country','Population'])                                 .T                                                              .stack(level=[0,1])                                             .reset_index()                                                  .rename(columns={'level_0':'date',
                                                   0:'vaccinated'},
                                                  )
pd_relational_model


# In[100]:


#pd_relational_model['single doss']=pd_relational_model['vaccinated']/2
#pd_relational_model['percent vaccination']=pd_relational_model['single doss']/pd_relational_model['Population']
pd_relational_model


# In[101]:


pd_relational_model.dtypes


# In[102]:


pd_relational_model['date']=pd_relational_model.date.astype('datetime64[ns]')


# In[ ]:


pd_relational_model.dtypes


# In[103]:


df_plot.to_csv('../ads_covid-19/data/processed/COVID_relational_confirmed_vaccinated.csv')


# In[104]:


data_path='../ads_covid-19/data/processed/time_series_covid19_vaccine_small_flat_table'
pd_raw=pd.read_csv(data_path)
pd_raw.head(15)


# In[105]:


plt.figure();
ax=df_plot.iloc[100:,:].set_index('date').plot()
#plt.ylim(10,30000)
ax.set_yscale('log')


# # plot.ly

# In[106]:


import plotly.graph_objects as go


# In[107]:


import plotly
plotly.__version__


# In[108]:


fig=go.Figure()


# In[110]:


for each in country_list:
    fig.add_trace(go.Scatter(x=df_plot.date,
                            y=df_plot[each],
                            #mode='markers',
                            #opacity=0.9,
                            line_width=2,
                            #marker_size=4,
                             name=each))

fig.update_layout(
    width=900,
    height=600,
    xaxis_title='Time',
    yaxis_title='Confirmed vaccinated people based on John Hopkins website'
    )
#fig.update_yaxes(type='linear',range=[0,1000000])
fig.update_layout(xaxis_rangeslider_visible=True)
fig.show()
fig.show(renderer='chrome')


# In[ ]:




