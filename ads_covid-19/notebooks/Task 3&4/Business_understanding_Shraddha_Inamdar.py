#!/usr/bin/env python
# coding: utf-8

# In[8]:


get_ipython().system('ls')


# In[9]:


import matplotlib.pyplot as mpl
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns

import matplotlib.image as mpimg
img = mpimg.imread('CRISP_DM.png')
imgplot = plt.imshow(img)
plt.show()
get_ipython().run_line_magic('matplotlib', 'inline')

mpl.rcParams['figure.figsize']=(16,9)
pd.set_option('display.max_rows',500)
sns.set(style='darkgrid')


# In[10]:


get_ipython().system('[CRISP_DM](../Applied data science/CRISP_DM.png)')


# John hopkins https://github.com/CSSEGISandData/COVID-19

# GITHUB csv data
# 
# git clone/pull https://github.com/CSSEGISandData/COVID-19.git

# In[11]:


import subprocess
import os
git_pull = subprocess.Popen( "/usr/bin/git pull" , 
                     cwd = os.path.dirname( '../ads_covid-19/data/raw/COVID-19/' ), 
                     shell = True, 
                     stdout = subprocess.PIPE, 
                     stderr = subprocess.PIPE )
(out, error) = git_pull.communicate()


print("Error : " + str(error)) 
print("out : " + str(out))


# In[12]:


pd.set_option('display.max_rows',500)
data_path = '../ads_covid-19/data/raw/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
pd_raw=pd.read_csv(data_path)
pd_raw.head()
pd_data_base= pd_raw.rename(columns={'Country/Region':'country','Province/State':'state'})
pd_data_base=pd_data_base.drop(['Lat','Long'],axis=1)
pd_data_base.head()


# In[13]:


test_pd=pd_data_base.set_index(['state','country']).T


# In[14]:


test_pd.columns


# In[15]:


test_pd.stack(level=0)


# In[16]:


test_pd.stack(level=[0,1]).reset_index()


# In[17]:


test_pd.dtypes


# In[1]:


#test_pd.dtypes['date']=test_pd.date.astype('dateime64[ns]')


# In[19]:


test_pd.dtypes


# In[20]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
df_plot=test_pd
df_plot.to_csv('../ads_covid-19/data/processed/COVID_relational_confirmed.csv',sep=';')


# In[21]:


pd_raw.columns[4:]


# In[22]:


time_idx=pd_raw.columns[4:]


# In[23]:


df_plot=pd.DataFrame({
    'date':time_idx
})
df_plot.head()


# In[24]:


pd_raw['Country/Region']


# In[25]:


pd_raw[pd_raw['Country/Region']=='US'].iloc[:,4::].sum(axis=0)[0:4]


# In[26]:


country_list=['Italy','US','Spain','Germany',]


# In[27]:


import numpy as np
for each in country_list:
    df_plot[each]=np.array(pd_raw[pd_raw['Country/Region']==each].iloc[:,4::].sum(axis=0))


# In[28]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure();
ax=df_plot.iloc[15:,4:].plot()
df_plot.plot()
plt.ylim(10,300000)


# Data type date

# In[29]:


df_plot.head()


# In[30]:


from datetime import datetime


# In[31]:


df_plot.date[0]


# In[32]:


time_idx=[datetime.strptime(each,"%m/%d/%y")for each in df_plot.date]


# In[33]:


time_str=[each.strftime('%Y-%m-%d') for each in time_idx]


# In[34]:


df_plot['date']=time_idx
type(df_plot['date'][0])


# In[35]:


df_plot.head()


# In[37]:


df_plot.to_csv('../ads_COVID-19/data/processed/COVID_small_flat_table.csv',sep=';',index=False)


# In[46]:


import pandas as pd
data_path='../ads_COVID-19/data/processed/COVID_small_flat_table.csv'
pd_raw=pd.read_csv(data_path)
pd_raw.head()


# In[38]:


#pd_data_base= pd_raw.rename(columns={'Country/Region':'country','Province/State':'State'})


# # Web scrapping

# In[66]:


import requests
from bs4 import BeautifulSoup
page = requests.get("https://www.rki.de/DE/Content/InfAZ/N/Neuartiges_Coronavirus/Fallzahlen.html")
soup=BeautifulSoup(page.content, 'html.parser')
soup.get_text()


# In[39]:


html_table=soup.find('table')
all_rows=html_table.find_all('tr')
final_data_list=[]


# In[48]:


for pos, rows in enumerate(all_rows):
    #print(pos)
    #print(rows)
    col_list=[(each_col.get_text(strip=True))for each_col in rows.find_all('td')]
    final_data_list.append(col_list)
pd_daily_status=pd.DataFrame(final_data_list).dropna().rename(columns={0:'state',1:'cases',2:'changes',3:'cases_per_100k',4:'7_day_incidence',5:'fatal'})
pd_daily_status.head()
    


# # REST API calls

# In[50]:


data=requests.get('https://services7.arcgis.com/mOBPykOjAyBO2ZKk/arcgis/rest/services/Coronafälle_in_den_Bundesländern/FeatureServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=json')


# In[53]:


import json
json_object=json.loads(data.content)


# In[54]:


type(json_object)


# In[55]:


json_object.keys()


# In[60]:


full_list=[]
for pos, each_dict in enumerate (json_object['features'][:]):
                                full_list.append(each_dict['attributes'])
                                
                                


# In[68]:


#pd.DataFrame(full_list)
pd_full_list=pd.DataFrame(full_list)
pd_full_list.head()


# In[71]:


pd_full_list.to_csv('../ads_covid-19/data/raw/NPGEO/GER_state_data.csv',sep=';')


# # API access via REST service, e.g. USA data

# In[3]:


import requests
url_endpoint='https://coronavirus-smartable.p.rapidapi.com/stats/v1/US/'
headers= {'X-RapidAPI-Host': 'coronavirus-smartable.p.rapidapi.com',
    'X-RapidAPI-Key': '3396648dafmsh107f83dbdd6f6c1p1b08e0jsnd21b0a12c819',
         }
response = requests.get(url_endpoint,headers=headers)


# In[4]:


print(response)


# In[8]:


#response.content
import json
US_dict=json.loads(response.content)
with open ('../ads_covid-19/data/SMARTABLE/US_data.txt','w') as outfile:
    json.dump(US_dict, outfile, indent=2)

