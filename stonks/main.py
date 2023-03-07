# from jinja2 import Environment, PackageLoader, select_autoescape
# env = Environment(
#     loader=PackageLoader("yourapp"),
#     autoescape=select_autoescape()
# )

# import pandas as pd
# import matplotlib.pyplot as plt
from dash import Dash, html, dcc, Output, Input
import plotly.express as px
import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy as np
from sklearn.linear_model import LinearRegression
import os
# import tensorflow as tf

app = Dash(__name__)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
# df = pd.DataFrame({
#     "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
#     "Amount": [4, 1, 2, 2, 4, 5],
#     "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
# })

# list of name for all files in Data/Stocks
onlyfiles = [f for f in listdir('Data/Stocks') if isfile(join('Data/Stocks', f))]

# test panda on 'a' stock
data = pd.read_csv('Data/Stocks/a.us.txt')

# fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

# testing displaying open through all time for the stock with abb: "a"

# graph1 figure
fig2 = px.bar(data, x="Date", y="Open", barmode="group")

# MACHINE LEARNING YAY (Linear Regression)
#
# TRAINS FIRST 80% OF AVG STOCK PER DAY AGAINST THE AVG OF AVG OF THE NEXT 20% OF STOCK DATA
# NOTE: MAY NOT BE VERY SCIENTIFIC BUT ITS COOL USING MACHINE LEARNING ORIGINALLY
#

#LOAD DATA

# BAD DATA: PBIO.TXT ACCP.TXT AMRH.US.TXT

x = []
y = []
temp = []
print("training\n")
i = 0
for file in onlyfiles:
    temp = []
    path = 'Data/Stocks/' + file
    # print(path)
    # if not (path == 'Data/Stocks/pbio.us.txt' or path == 'Data/Stocks/accp.us.txt'or path == 'Data/Stocks/amrh.us.txt' or
    #         path == 'Data/Stocks/vist.us.txt' or path == 'Data/Stocks/srva.us.txt' or path == 'Data/Stocks/bbrx.us.txt' or
    #         path == 'Data/Stocks/bolt.us.txt' or path == 'Data/Stocks/amrhw.us.txt'):
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        dat = pd.read_csv(path)
        temp = dat['Open'].values.tolist()

    
    if len(temp) > 810:
        x.append(temp[-800:-202])
        y.append(temp[-201:-1])
    
    i += 1
    if i == 100:
        break # for testing purposes to reduce loadtime



# IMPORT NP ARRAY FOR X IS LIST OF LIST OF 80 ALL STOCK DATA
#       X = [**aapl avg for first 80%**, **googl avg for first 80%**]
#       Y = [avg(aapl for last 20%), avg(googl for last 20%)]
print("TRAINING DONE")
train_y = []
for ys in y:
    train_y.append(sum(ys) / len(ys))

print(x[0])
print(train_y[0])

## regress
reg = LinearRegression().fit(x,train_y)
print("FINISHED REGRESSION")

## predict

# main website program
app.layout = html.Div(children=[
    #header
    html.Div(children = 'Stocks, an interactable website for everything stocks',
             className = 'header'),
    # basic display
    html.Div(children=[
        html.H1(children='Basic Stock Display From All Stocks 11/18/1999 - 11/10/2017'),

        # html.Div(children='''
        #     Dash: A web application framework for your data.
        # '''),

        html.Div(children=[
            html.Div(html.Label('Select Stock and Metric to Dispay on Y axis over Time')),
            html.Div(dcc.Dropdown(onlyfiles,id='file',value=onlyfiles[0]),className="drop2"),
            html.Div(dcc.Dropdown(data.columns[1:],id='test1',value = data.columns[1]),className="drop2"),
            

            # html.Br(),
            # html.Label('Multi-Select Dropdown'),
            # dcc.Dropdown(['New York City', 'Montréal', 'San Francisco'],
            #              ['Montréal', 'San Francisco'],
            #              multi=True),

            # html.Br(),
            # html.Label('Radio Items'),
            # dcc.RadioItems(['New York City', 'Montréal', 'San Francisco'], 'Montréal'),
        ], style={'padding': 10, 'flex': 1}),

        html.Div(dcc.Graph(
            id='example-graph',
            figure=fig2
        ),className="graph1"),
    ],className="metric"),
    html.Div(children=[
        html.H1(children='Calculators'),

        html.H2(children='Average Stock Price Between Dates',),
        html.Div(children = 
            [html.Div(dcc.Dropdown(onlyfiles,id='file2',value=onlyfiles[0]),className="drop5"),
            html.Div(dcc.Dropdown(data.columns[1:],id='test2',value = data.columns[1]),className="drop5"),
            html.Div(dcc.Dropdown(['Average','Max','Min'],id='stat',value='Average'),className="drop5"),
            html.Div(dcc.Dropdown([1999,2000,2001,2002,2003,2004,2005,2006,2007,
                                   2008,2009,2010,2011,2012,2013,2014,2015,2016],id='beginDate',value = 1999),className="drop5"),
            html.Div(dcc.Dropdown([2000,2001,2002,2003,2004,2005,2006,2007,
                                   2008,2009,2010,2011,2012,2013,2014,2015,2016,2017],id='endDate',value = 2017),className="drop5")]),

        html.H1(id="result")

    ],id="section"),
    html.Div(children=[html.H1(children='Machine Learning: Selected which date to start at to predict average stock price for next x years'),
                      # dcc.Dropdown([2000,2001,2002,2003,'Present'],id='train date',value = 2010),
                      #  dcc.Dropdown([1,2,3,4,5],id='predict date',value = 5),
                        dcc.Dropdown(onlyfiles,id='trainstock',value = onlyfiles[0]),
                        html.H1(id="pred")],
             className = "stocks")
],style={'font':'verdana',
         'margin':0})

# callbacks
@app.callback(
#outputs
    # graph
    Output(component_id='example-graph', component_property='figure'),

    # stat
    Output(component_id='result',component_property='children'),

    #ML
    Output(component_id='pred', component_property = 'children'),

#inputs
    # graph
    Input(component_id='test1', component_property='value'),
    Input(component_id='file',component_property='value'),

    #stat
    Input(component_id='test2', component_property='value'),
    Input(component_id='file2',component_property='value'),
    Input(component_id='stat',component_property='value'),
    Input(component_id='beginDate',component_property='value'),
    Input(component_id='endDate',component_property='value'),

    # ML
    Input(component_id = 'trainstock', component_property = 'value')
    
)
# can name func anything, updates callbacks
def update_graph898(test1,file,
                    test2,file2,stat,beginDate,endDate,
                    trainstock):
    #graph
    path = 'Data/Stocks/' + file
    data = pd.read_csv(path)
    fig2 = px.bar(data, x="Date", y=test1, barmode="group")
    fig2.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    font_color='black'
    )

    #stat
    path2 = 'Data/Stocks/' + file2
    data2 = pd.read_csv(path2)
    #date index accurate outside 2002-2014
    # NOTE CODE DOES NOT REALLY WORK PROPERLY FOR DATE AND TIME
    # !!!!!!!!!!!!!
    #!!!!!!!!!!!!!!!!!!!!!

    date_index = {1999:0,2000:30,2001:529,2002:778,2003:1027,2004:1276,2005:1525,2006:1774,2007:1900,2008:2100,
                  2009:2300,2010:2500,2011:2700,2012:2900,2012:3100,2013:3300,2014:3500,2015:3799,2016:4051,2017:4303,2018:4520}
    # newdata = data2.index.searchsorted('Date')
    # dateq = '"' + str(beginDate) + '-1-01"' + ' < index <= "' + str(endDate) + '-12-31"'


    # datatime = data2.query(dataq)

    begin = int(beginDate)
    end = int(endDate)

    datatime = data2[date_index[begin]:date_index[end+1]]

    # datatime = data2.loc[int((data2['Date'][:3]) >= int(beginDate)) & (int(data2['Date'][:3]) <= int(endDate))]
    returnstat=""
    if stat == "Average":
        returnstat=sum(datatime[test2]) / len(datatime[test2])
    elif stat == "Max":
        returnstat = max(datatime[test2])
    else:
        returnstat = min(datatime[test2])

    returnstat = "Result: " + str(returnstat)
    # + "length" + str(len(datatime[test2])) + "3wijf"+  str(date_index[begin]) + "fjoiewjfoijawe " + str(date_index[end+1])


    # prediction for ML
    path3 = 'Data/Stocks/' + trainstock
    preddata = pd.read_csv(path3)
    predx = preddata['Open'].values.tolist()
    predx = predx[-599:-1]
    
    if len(temp) > 810:
        x.append(temp[-800:-202])
        y.append(temp[-201:-1])

    predx = np.reshape(predx,(1,-1))
    r = reg.predict(predx)


    # returns all outputs
    return fig2, returnstat, r

if __name__ == '__main__':
    app.run_server(debug=True)

app = Dash(__name__)

# def imp(path_to_file):
#     # importing dataset

#     f = open(path_to_file,'r')
#     data = f.readlines()


#     f.close()

#     return data

# if __name__ == "__main__":
#     file = "Data/Stocks/a.us.txt"
#     data = imp(file)

#     labels = data[0][:-1] # to get rid of newline

#     info = data[1:]

#     labels = labels.split(",")

#     print(labels)

############
############
############
# TO DO LIST!!
# Incorporate date in the calculators
# Do the CSS styling
# Add more interactablitity in the graphs
# Make the data live updated with time
# Add different webpages
# Add analysis
# Allow user to make their own regressions
# Machine learning between predicted stocks?
#     trainX = Previous stock data, trainY = next day's stock data?
#     train last X years minimume 5, tpo average stock Y in next year, to give prediction of stock in next year