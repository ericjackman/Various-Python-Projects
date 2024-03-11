from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import pandas as pd
from dash import Dash, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from statsmodels.tsa.ar_model import AutoReg

app = Dash(__name__)


# Creates a banner at the top of the page
def build_banner():
	return html.Div(
		id="banner",
		className="banner",
		children=[
			html.Div(
				id="banner-text",
				children=[
					html.H1("Reddit Data Analysis"),
					html.H4("Pre-Processing and Reporting"),
					html.H4("Eric Jackman and Kristoffe Wilson")
				],
			)
		],
	)


# Creates tabs to navigate the page
def build_tabs():
	return html.Div(
		id="tabs",
		className="tabs",
		children=[
			dcc.Tabs(
				id="tabs-with-classes",
				value='tab-1',
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
				])
		])


# Changes content when user switches tabs
@app.callback(Output('tabs-content-classes', 'children'),
			  Input('tabs-with-classes', 'value'))
def render_content(tab):
	if tab == 'tab-1':
		return html.Div([
			html.H2(className="small_div", children='Pre-Process'),
			pre_process(),
			dbc.Alert(id='cell_out')
		])
	elif tab == 'tab-2':
		return html.Div([
			html.H2('GoogleApp Data')
		])
	elif tab == 'tab-3':
		return html.Div([
			html.H2('Reddit Data'),
			html.H3(className="small_div", children='Time Series Analysis:'),
			reddit()
		])


# Creates a table to be displayed from a dataframe
def create_table(dataframe):
	return dash_table.DataTable(data=dataframe.to_dict('records'),
								columns=[{"name": i, "id": i} for i in dataframe.columns],
								page_size=10,
								css=[{'selector': 'table', 'rule': 'table-layout: fixed'}],
								style_cell={
									'width': '{}%'.format(len(dataframe.columns)),
									'textOverflow': 'ellipsis',
									'overflow': 'hidden',
									'textAlign': 'left'
								}
								)


# Creates content for the pre-process tab
def pre_process():
	# Import data to dataframes
	redditDF = pd.read_csv('DSC496_2022_Spring_Reddit_Fidelity.csv')
	googleDF = pd.read_csv('DSC311Spring2022_TermProject_GoogleAppReviews_v1.0.0.csv')

	# Clean reddit date column (YYYY-MM)
	split_date = redditDF.created_utc.str.split('-', expand=True)
	split_date.columns = ["year", "month", "day"]
	monthlyDF = redditDF[['created_utc', 'score']]
	monthlyDF['month'] = split_date['year'] + '-' + split_date['month']
	monthlyDF = monthlyDF.groupby(by=['month'], as_index=False).sum()

	# Create div with tables
	return html.Div(
		id="pre_process_div",
		children=[
			html.Div(
				id="data_table",
				children=[
					html.Div(
						id="table",
						children=[
							html.H4('Reddit Data'),
							create_table(redditDF),
							html.H4('Processed Reddit Data'),
							create_table(monthlyDF),
							html.H4('Google Data'),
							create_table(googleDF)
						])
				])
		])


# Creates content for the reddit tab
def reddit():
	# Import data to dataframe
	df = pd.read_csv("DSC496_2022_Spring_Reddit_Fidelity.csv")

	# Clean date column (YYYY-MM)
	df['month'] = pd.to_datetime(df['created_utc']).dt.to_period('M')

	# Create new dataframe for month and score per month
	monthlyDF = df[['month', 'score']]
	monthlyDF = monthlyDF.groupby(by=['month'], as_index=False).sum()
	monthlyDF.set_index('month', inplace=True)

	# Plot score per month
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=monthlyDF.index.to_timestamp(), y=monthlyDF['score'],
							 mode='lines+markers',
							 name='Score Per Month'))

	fig.update_layout(
		title="Scores Per Month",
		xaxis_title="Month",
		yaxis_title="Score"
	)

	graph1 = dcc.Graph(
		id='TSA1',
		figure=fig,
		style={'height': '600px'}
	)

	# Create testing and training sets for auto-regression model
	train, test = train_test_split(df["score"], test_size=0.2)

	# Train auto-regression model
	model = AutoReg(train, lags=1000)
	model_fit = model.fit()

	# Make predictions using auto-regression model
	predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

	# Plot auto-regression predictions
	fig2 = go.Figure()
	fig2.add_trace(go.Scatter(x=monthlyDF.index.to_timestamp(), y=test.values,
							  mode='lines+markers',
							  name='Expected'))
	fig2.add_trace(go.Scatter(x=monthlyDF.index.to_timestamp(), y=predictions.values,
							  mode='lines+markers',
							  name='Predicted'))

	fig2.update_layout(
		title="Auto-Regression Model",
		xaxis_title="Month",
		yaxis_title="Score"
	)

	graph2 = dcc.Graph(
		id='TSA2',
		figure=fig2,
		style={'height': '600px'}
	)

	# Calculate 6 and 12 month rolling averages
	monthlyDF['6_month_MA'] = monthlyDF.score.rolling(6, min_periods=1).mean()
	monthlyDF['12_month_MA'] = monthlyDF.score.rolling(12, min_periods=1).mean()

	# Plot 6 and 12 month rolling averages
	fig3 = go.Figure()
	fig3.add_trace(go.Scatter(x=monthlyDF.index.to_timestamp(), y=monthlyDF['score'],
							  mode='lines+markers',
							  name='Score Per Month'))
	fig3.add_trace(go.Scatter(x=monthlyDF.index.to_timestamp(), y=monthlyDF['6_month_MA'],
							  mode='lines+markers',
							  name='MA_6'))
	fig3.add_trace(go.Scatter(x=monthlyDF.index.to_timestamp(), y=monthlyDF['12_month_MA'],
							  mode='lines+markers',
							  name='MA_12'))
	fig3.update_layout(
		title="6 and 12 Month Moving Averages",
		xaxis_title="Month",
		yaxis_title="Score"
	)

	graph3 = dcc.Graph(
		id='TSA3',
		figure=fig3,
		style={'height': '600px'}
	)

	# Calculate exponential smoothing moving average
	monthlyDF['EMA_0.1'] = monthlyDF.score.ewm(alpha=0.1, adjust=False).mean()
	monthlyDF['EMA_0.3'] = monthlyDF.score.ewm(alpha=0.3, adjust=False).mean()

	# Plot exponential smoothing moving average
	fig4 = go.Figure()
	fig4.add_trace(go.Scatter(x=monthlyDF.index.to_timestamp(), y=monthlyDF['score'],
							  mode='lines+markers',
							  name='Score Per Month'))
	fig4.add_trace(go.Scatter(x=monthlyDF.index.to_timestamp(), y=monthlyDF['EMA_0.1'],
							  mode='lines+markers',
							  name='EMA_0.1'))
	fig4.add_trace(go.Scatter(x=monthlyDF.index.to_timestamp(), y=monthlyDF['EMA_0.3'],
							  mode='lines+markers',
							  name='EMA_0.3'))

	fig4.update_layout(
		title="Exponential Smoothing",
		xaxis_title="Month",
		yaxis_title="Score"
	)

	graph4 = dcc.Graph(
		id='TSA4',
		figure=fig4,
		style={'height': '600px'}
	)

	# Return div containing the graphs
	return html.Div(
		id="reddit-tsa-graphs",
		children=[
			graph1,
			graph2,
			graph3,
			graph4]
	)


if __name__ == '__main__':
	app.layout = html.Div(
		id="big-app-container",
		children=[
			build_banner(),
			html.Div(
				id="app-container",
				children=[
					build_tabs(),
					html.Div(id='tabs-content-classes'),
				],
			)
		],
	)
	app.run_server(debug=True)
