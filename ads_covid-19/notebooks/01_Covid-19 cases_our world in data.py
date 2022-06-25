#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


mpl.rcParams['figure.figsize']=(16,9)
pd.set_option('display.max_rows',500)
sns.set(style='darkgrid')


# In[3]:


data_path='../ads_covid-19/data/raw/COVID-19/owid-covid-data.csv'
pd_raw=pd.read_csv(data_path)


# In[4]:


pd_raw.head()


# # GITHUB csv data
# git clone/pull https://github.com/CSSEGISandData/COVID-19.git

# In[6]:


git_pull = subprocess.Popen( "/usr/bin/git pull" , 
                     cwd = os.path.dirname( '../ads_covid-19/data/raw/COVID-19/time_series_covid19_vaccine_doses_admin_global.csv'), 
                     shell = True, 
                     stdout = subprocess.PIPE, 
                     stderr = subprocess.PIPE )
(out, error) = git_pull.communicate()


print("Error : " + str(error)) 
print("out : " + str(out))


# In[7]:


df_new_list=pd_raw[['date','location','population','total_cases']]
df_new_list


# In[8]:


df_new_list['absolute cases per population size']=df_new_list['total_cases']/df_new_list['population']
df_new_list


# In[29]:


#country_list=['India','Germany','Canada','Spain','Italy']
#df_india=df_new_list[df_new_list['location']== 'India']
#df_Germany=df_new_list[df_new_list['location']== 'Germany']


# In[30]:


df_germany   = df_new_list.loc[:, 'location'].str.contains('Germany')
df_Italy    = df_new_list.loc[:, 'location'].str.contains('Italy')
df_France    = df_new_list.loc[:, 'location'].str.contains('France')
df_Israel    = df_new_list.loc[:, 'location'].str.contains('Israel')
df_Latvia   = df_new_list.loc[:, 'location'].str.contains('Latvia')
df_combined=df_new_list.loc[df_germany | df_Italy | df_France | df_Israel | df_Latvia, :]


# In[31]:


import plotly.express as px
fig=px.line(df_combined,x="date",y="absolute cases per population size",color='location',title='Distribution of Covid-19 cases over time, data source : https://ourworldindata.org/')
#fig=px.line(df_Germany,x="date",y="percent_vaccination")
fig.update_yaxes(type='linear')
fig.update_layout(xaxis_title='Date',
yaxis_title='Absolute covid-19 cases per population size as per https://ourworldindata.org/',
width=1200,
height=800,)
fig.show()
fig.show(renderer='chrome')


# In[ ]:





# In[ ]:




