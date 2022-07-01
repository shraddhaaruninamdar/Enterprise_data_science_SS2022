#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from datetime import datetime
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns


# In[2]:


mpl.rcParams['figure.figsize']=(20,16)
pd.set_option('display.max_rows',500)
sns.set(style='darkgrid')


# In[3]:


#!.[CRISP_DM].(../ads_covid-19/reports/figures/CRISP_DM.png)


# John hopkins https://github.com/CSSEGISandData/COVID-19

# GITHUB csv data
# 
# git clone/pull https://github.com/CSSEGISandData/COVID-19.git

# In[4]:


df_plot=pd.read_csv('../Applied data science/COVID_small_flat_table.csv',sep=';')
country_list=['Italy','US','Spain','Germany',]
df_plot.head()


# In[5]:


plt.figure();
ax=df_plot.iloc[15:,:].set_index('date').plot()
plt.ylim(10,30000)
ax.set_yscale('log')


# # Plot.ly

# In[6]:


import plotly.graph_objects as go


# In[7]:


import plotly
plotly.__version__


# In[9]:


fig=go.Figure()
for each in country_list:
    fig.add_trace(go.Scatter(x=df_plot.date,
                                y=df_plot[each],
                                mode='markers+lines',
                                opacity=0.9,
                                line_width=2,
                                marker_size=4,
                                name=each
                                    )
                             )
fig.update_layout(
    width=1600,
    height=1200,
    xaxis_title="Time",
    yaxis_title="Confirmed infected people (source johns hopkins case, log-scale)",
)
fig.update_yaxes(type='log',range=[1.1,8.8])

fig.update_layout(xaxis_rangeslider_visible=True)
fig.show(renderer='chrome')


# In[9]:


import dash
import dash_core_components as doc
import dash_html_components as html

app = dash.Dash()
app.layout = html.Div([
    html.Label('Multi-Select Country'),
    doc.Dropdown(
        id='country_drop_down',
        options=[
            {'label':'Italy','value':'Italy'},
            {'label':'US','value':'US'},
            {'label':'Spain','value':'Spain'},
            {'label':'Germany','value':'Germany'}
        ],
        value=['US','Germany'],
        multi=True
    ),
    doc.Graph(figure=fig,id='main_window_slope')
])


# In[10]:


from dash.dependencies import Input, Output

@app.callback(
    Output('main_window_slope','figure'),
    [Input('country_drop_down','value')])
def update_figure(country_list):
    
    traces=[]
    for each in country_list:
        traces.append(dict(x=df_plot.date,
                           y=df_plot[each],
                           mode='markers+lines',
                           opacity=0.9,
                           line_width=2,
                           marker_size=4,
                           name=each
                    )
                )
    return{
        'data':traces,
        'layout':dict(
            width=1280,
            height=720,
            xaxis_title="Time",
            yaxis_title="confirmed infected people (source johns hopkins case, log-scale)",
            xaxis={'tickangle':-45,
                  'nticks':20,
                  'tickfont':dict(size=14,color="#7f7f7f"),
                  },
            yaxis={'type':"log",
                  'range':'[1.1,5.5]'
                  }
        )
    }


# In[ ]:


app.run_server(debug=True,use_reloader=False)


# In[ ]:




