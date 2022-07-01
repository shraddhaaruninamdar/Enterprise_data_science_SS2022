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


data_path='../ads_covid-19/data/processed/COVID_vaccinated_percent.csv'
df_new_list=pd.read_csv(data_path)


# ###### The .CSV file owid-covid-data.csv is of size 52 MB and therfore do not get uploaded to Git. The data extracted is saved in the CSV file COVID_vaccinated_percent.csv and that file is uploaded to Git. A separate program with smaller CSV is prepared and saved as Vaccination rate_our world in data-with small data set.ipynb

# In[4]:


df_new_list.head()


# # GITHUB csv data
# git clone/pull https://github.com/CSSEGISandData/COVID-19.git

# In[5]:


git_pull = subprocess.Popen( "/usr/bin/git pull" , 
                     cwd = os.path.dirname( '../ads_covid-19/data/raw/COVID-19/time_series_covid19_vaccine_doses_admin_global.csv'), 
                     shell = True, 
                     stdout = subprocess.PIPE, 
                     stderr = subprocess.PIPE )
(out, error) = git_pull.communicate()


print("Error : " + str(error)) 
print("out : " + str(out))


# In[6]:


df_germany   = df_new_list.loc[:, 'location'].str.contains('Germany')
df_Italy    = df_new_list.loc[:, 'location'].str.contains('Italy')
df_France    = df_new_list.loc[:, 'location'].str.contains('France')
df_Israel    = df_new_list.loc[:, 'location'].str.contains('Israel')
df_Latvia   = df_new_list.loc[:, 'location'].str.contains('Latvia')
df_combined=df_new_list.loc[df_germany | df_Italy | df_France | df_Israel | df_Latvia, :]


# ### In 2 figures below, the first one is linear plot and the second one is logarithmic. In this particular case, linear plots make it easier to understand the data therefore figure with linear scale is uploaded to OLAT.

# In[11]:


import plotly.express as px
fig=px.line(df_combined,x="date",y="percent_vaccination",color='location',range_x=['2020-09-01','2022-06-01'],title='Vaccination rate in percent over time, Data source: https://ourworldindata.org/')
#fig=px.line(df_Germany,x="date",y="percent_vaccination")
fig.update_yaxes(type='linear',range=[0,1])
fig.update_layout(xaxis_title='Date',
yaxis_title='Fully vaccinated people in percent as per https://ourworldindata.org/',
width=1000,
height=800,)
fig.show()
fig.show(renderer='chrome')


# In[10]:


import plotly.express as px
fig=px.line(df_combined,x="date",y="percent_vaccination",color='location',range_x=['2020-09-01','2022-06-01'],title='Vaccination rate in percent over time, Data source: https://ourworldindata.org/')
#fig=px.line(df_Germany,x="date",y="percent_vaccination")
fig.update_yaxes(type='log')
fig.update_layout(xaxis_title='Date',
yaxis_title='Fully vaccinated people in percent as per https://ourworldindata.org/',
width=1000,
height=800,)
fig.show()


# In[ ]:




