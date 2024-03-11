#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install plotly
#pip install dash


# In[8]:


#!/usr/bin/env python
# coding: utf-8

# In[9]:


from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, dash_table
import dash_bootstrap_components as dbc
import matplotlib.pyplot as plt
from plotly.tools import mpl_to_plotly
import plotly.graph_objects as go


# In[9]:






app = Dash(__name__)

df = pd.read_csv('DSC496_2022_Spring_Reddit_Fidelity.csv')

    

fig1 = px.scatter(df, x="score", y="created_utc",
                 size="num_comments", color="num_comments", hover_name="title",
                 log_x=True, size_max=60)

fig2 = px.scatter(df, x="score", y="num_comments")

#df = df[0:2000]
def reddit_TSA():
    data=pd.read_csv('DSC496_2022_Spring_Reddit_Fidelity.csv')
    split_date = data.created_utc.str.split('-', expand=True)
    split_date.columns =["year","month","day"]

# Group by month and find mean score
    data['month'] = split_date.year.str[2:] + '/' + split_date['month']
    perMonth = data.groupby(['month']).agg({'score': ['mean']})
    data['perMonth'] = perMonth
    maxScore = data.groupby(['month']).agg({'score': ['max']})
    print(perMonth.iloc(1))

    data['MA_6'] = data.score.rolling(6, min_periods=1).mean()
    data['MA_12'] = data.score.rolling(12, min_periods=1).mean()
    print(data[['score', 'MA_6', 'MA_12']])

    colors = ['cyan', 'red', 'orange']
# Line plot

    # Create traces
    fig = go.Figure()
    fig.add_trace(go.Scatter( x=data['month'], y=data["perMonth"],
                        mode='lines+markers',
                        name='perMonth'))
    fig.add_trace(go.Scatter(x=data['month'], y=data["MA_6"],
                         mode='lines+markers',
                         name='MA_6'))
    fig.add_trace(go.Scatter(x=data['month'], y=data["MA_12"],
                         mode='lines+markers', name='MA_12'))
    #fig.show()
    return fig

    '''
    data.plot(x='month',y=["perMonth","MA_6","MA_12"], color=colors, linewidth=3, figsize=(12,6))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.legend(labels =['Average score', '6-month MA', '12-month MA'], fontsize=14)

    plt.title('The monthly average score', fontsize=20)
    plt.xlabel('Month', fontsize=16)
    plt.ylabel('Temperature [°C]', fontsize=16)
    fig3=plt
    return fig3
    '''
#fig3.show()

#define the components
#banner
def build_banner():
    return html.Div(
        id="banner",
        className="banner",
        children=[
            html.Div(
                id="banner-text",
                children=[
                    html.H5("Reddit Data Analysis"),
                    html.H6("Pre-Processing and Reporting"),
                ],
            )
        ],
    )

#tabs
def build_tabs():
    return html.Div(
        id ="tabs",
        className="tabs",
        children=[
    dcc.Tabs(
        id="tabs-with-classes",
        value='tab-1',
        #parent_className='custom-tabs',
        className='custom-tabs',
        children=[
            dcc.Tab(
                id="tab-1",
                label='Pre-Process',
                value='tab-1',
                className='custom-tab',
                selected_className='custom-tab--selected'
            ),
            dcc.Tab(
                id="tab-2",
                label='GoogleApp Data',
                value='tab-2',
                className='custom-tab',
                selected_className='custom-tab--selected'
            ),
            dcc.Tab(
                id="tab-3",
                label='Reddit Data',
                value='tab-3', className='custom-tab',
                selected_className='custom-tab--selected'
            ),
            dcc.Tab(
                id="tab-4",
                label='Total Data',
                value='tab-4',
                className='custom-tab',
                selected_className='custom-tab--selected'
            ),
        ])
])

app.layout = html.Div(
            id="big-app-container",
            children=[
                build_banner(),
                html.Div(
                    id="app-container",
                    children=[
                        build_tabs(),
                        # Main app
                        html.Div(id='tabs-content-classes'),
                    ],
                )
                #,dcc.Store(id="value-setter-store", data=init_value_setter_store()),
                #dcc.Store(id="n-interval-stage", data=50),
                #generate_modal(),
            ],
        )

#figures and tables for the pre_process
def pre_process():
    return html.Div(
        id="pre_process_div",
        children=[
            html.Div(
                id="data_table",
                children=[
                    html.Div(
                        id="table",
                        children=[
                               generate_dataTable(df)
                            ]),
                    
                    html.Div(
                        id="graph",
                        children=[
                                dcc.Graph(
                                id='comments2',
                                figure=fig2,
                                style={'height': '300px'}
                                )]
                    )
            ]),
            html.Div(
                id="scatter_chart",
                children=[
                    html.Div([
                        dcc.Graph(
                        id='comments1',
                        figure=fig1
                        )
                    ])
            ]),
        ])
def reddit():
    return html.Div(
        id="reddit_tsa_div",
        children=[
            html.Div(
                id="data_table",
                children=[
                    html.Div(
                        id="graph",
                        children=[
                                dcc.Graph(
                                id='TSA',
                                figure=reddit_TSA(),
                                style={'height': '300px'}
                                )]
                    )
            ])
        ])




def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])
'''tooltip_data=[
                                {
                                    column: {'value': str(value), 'type': 'markdown'}
                                    for column, value in row.items()
                                } for row in df.to_dict('records')]
                                
    tooltip_conditional=[
        {
            'if': {'filter_query': '{value_length} > 100'
                   .format(value_length = len(str(value)))}
                ,'value': str(value), 'type': 'markdown'

        } for row in df.to_dict('records') for column, value in row.items()
    ]'''
def generate_dataTable(dataframe):
    return dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns],id="dataset"
                                ,fixed_columns={'headers':True,'data':1},fixed_rows={'headers':True,'data':0}
                                ,cell_selectable=True,page_size=50
                                ,style_table={'height': '500px','width':'500px', 'overflowY': 'auto'}
                                ,style_cell={
                                'overflow': 'hidden',
                                'textOverflow': 'ellipsis',
                                'minWidth': 100,
                                'maxWidth': 1000,}
                                ,tooltip_data=[{
                                        column: {'value': str(value), 'type': 'markdown'} 
                                    for column, value in row.items() if column in ['url','title','created_utc','selftext'] }
                                    for row in df.to_dict('records') 
                                    ]
                                ,css=[{
                                    'selector': '.dash-table-tooltip',
                                    'rule': 'background-color: grey; font-family: monospace; color: white'
                                }]
                                ,tooltip_duration=None
                                , style_header={
                                    'backgroundColor': 'rgb(210, 210, 210)',
                                    'color': 'black',
                                    'fontWeight': 'bold'
                                }
                                ,style_data_conditional=[
                                    {
                                        'if': {'row_index': 'odd'},
                                        'backgroundColor': 'rgb(220, 220, 220)',
                                    }
                                ],
                                style_data={
                                    'backgroundColor': 'white',
                                    'color': 'black'
                                },)



@app.callback(Output('tabs-content-classes', 'children'),
              Input('tabs-with-classes', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div(
            [
            html.H3(className="small_div",children='Pre-Process'),
            html.H4(children='Spring_Reddit_Fidelity'),
            pre_process(),
            dbc.Alert(id='cell_out'),
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H3('GoogleApp Data')
        ])
    elif tab == 'tab-3':
        return html.Div([
            html.H3('Reddit Data'),
            html.H3(className="small_div",children='Reddit_tsa'),
            html.H4(children='Spring_Reddit_Fidelity'),
            reddit(),
        ])
    elif tab == 'tab-4':
        return html.Div([
            html.H3('Total Data')
        ])

    
@app.callback(Output('cell_out', 'children'), Input('dataset', 'active_cell'))
def update_graphs(active_cell):
    return str(active_cell) if active_cell else "Click the table"
    
if __name__ == '__main__':
    app.run_server(debug=True)


# In[ ]:





# In[7]:


get_ipython().run_line_magic('tb', '')


# In[ ]:


# In[ ]:




