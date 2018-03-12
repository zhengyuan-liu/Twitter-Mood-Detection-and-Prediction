from math import log
import glob
import numpy as np
import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go

getDate = lambda x: x.date()
f = lambda x: log(x+1) if x>0 else -log(-x+1)

def load_single(filename, agg, l):
	df = pd.read_csv(filename, parse_dates=['dateTime'])
	start = filename.find('test_') + 5
	end = filename.find('_result')
	name = filename[start:end].title()
	df = df.rename(columns = {'prediction':name})
	if agg:
		df['dateTime'] = df['dateTime'].map(getDate)
		df = df.groupby('dateTime').sum()
	else:
		df = df.set_index('dateTime')
	if l: df[name] = df[name].map(f)
	return df

def load_multiple(files='../prediction/*.csv', agg=True, l=True):
	first = True
	df = None
	for filename in glob.iglob(files):
		df0 = load_single(filename, agg, l)
		if first:
			df = df0
			first = False
		else:
			df = df.join(df0, how='outer', sort=True)
	return df

aggregate_by_date = True
log_value = False
df = load_single('../prediction/test_trump_result.csv', aggregate_by_date, log_value)
# df = load_multiple()

traces_line = []

for col in df.columns:
	traces_line.append(go.Scatter(
		x = df.index,
		y = df[col],
		name = col,
		connectgaps = True
	))

layout_line = go.Layout(
    title='Twitter Users Happiness Line Chart',
    yaxis=dict(title='Happiness Index'),
    xaxis=dict(title='Date Time')
)

fig_line = go.Figure(data=traces_line, layout=layout_line)

py.plot(fig_line, filename='line_chart.html')

traces_heat = [
	go.Heatmap(
		z = df.T.values.tolist(),
		x = df.index,
		y = df.columns,
		# colorscale = 'Viridis'
		# colorscale=[[0.0, 'rgb(146,168,209)'], [0.1111111111111111, 'rgb(168,185,218)'], [0.2222222222222222, 'rgb(190,203,227)'], [0.3333333333333333, 'rgb(211,220,237)'], [0.4444444444444444, 'rgb(233,238,246)'], [0.5555555555555556, 'rgb(253,244,244)'], [0.6666666666666666, 'rgb(252,234,233)'], [0.7777777777777778, 'rgb(250,223,223)'], [0.8888888888888888, 'rgb(249,213,212)'], [1.0, 'rgb(247,202,201)']]
		colorscale=[[0.0, 'rgb(146,168,209)'], [0.25, 'rgb(201,212,232)'], [0.75, 'rgb(251,229,228)'], [1.0, 'rgb(247,202,201)']]
		# colorscale=[[0.0, 'rgb(201,212,232)'], [0.6666666666666666, 'rgb(251,229,228)'], [1.0, 'rgb(247,202,201)']]
	)
]

layout_heat = go.Layout(
    title='Twitter Users Happiness Heatmap',
    xaxis = dict(title='Date Time'),
    yaxis = dict(title='Users', ticks='')
)

fig_heat = go.Figure(data=traces_heat, layout=layout_heat)

py.plot(fig_heat, filename='heatmap.html')
