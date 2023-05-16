from flask import Flask, render_template, request
import plotly
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import requests
from tensorflow import keras
from tensorflow.keras.models import load_model

app = Flask(__name__)
btc_model = keras.models.load_model('lstm_model.h5')
eth_model = keras.models.load_model('lstm1_model.h5')
doge_model = keras.models.load_model('DOGE_model.h5')

btc_model.compile(loss='mse', optimizer='adam', run_eagerly=True)
eth_model.compile(loss='mse', optimizer='adam', run_eagerly=True)
doge_model.compile(loss='mse', optimizer='adam', run_eagerly=True)


@app.route('/lstm-btc-prediction')
def lstm_prediction():
    
    btc_url = 'https://api.pro.coinbase.com/products/BTC-USD/ticker'
    btc_response = requests.get(btc_url).json()
    btc_current_price = float(btc_response['price'])

    # Load historical data for BTC prediction
    btc_df = yf.download('BTC-USD', period='1d', interval='1m')
    
    btc_data = btc_df['Close'].values.reshape(-1, 1)

    # Normalize the BTC data
    btc_scaler = MinMaxScaler()
    btc_scaled_data = btc_scaler.fit_transform(btc_data)

    # Prepare the BTC data for prediction
    n_steps = 60
    btc_X = []
    for i in range(n_steps, len(btc_scaled_data)):
        btc_X.append(btc_scaled_data[i - n_steps:i, 0])
    btc_X = np.array(btc_X)

    # Make the BTC prediction
    btc_y_pred = btc_model.predict(btc_X)
    btc_y_pred = btc_scaler.inverse_transform(btc_y_pred)
    # Determine the BTC recommendation and predicted price
    if btc_y_pred[-1] > btc_current_price:
        btc_recommendation = 'BUY'
    elif btc_y_pred[-1] < btc_current_price:
        btc_recommendation = 'SELL'
    else:
        btc_recommendation = 'HOLD'
    btc_predicted_price = btc_y_pred[-1][0]

    

    
    # Fetch live trade data from Coinbase Pro API for doge
    

    # Determine the doge recommendation and predicted price
    
    # Create the plotly chart for BTC
    btc_trace1 = go.Scatter(x=btc_df.index, y=btc_df['Close'], mode='lines', name='Current Price')
    btc_trace2 = go.Scatter(x=btc_df.index[n_steps:], y=btc_y_pred[:, 0], mode='lines', name='Predicted Price')

    btc_layout = go.Layout(title='BTC Price Prediction', xaxis_title='Date', yaxis_title='Price', legend=dict(x=0, y=1))

    btc_fig = go.Figure([btc_trace1, btc_trace2], layout=btc_layout)

   

# Convert the Plotly figures to JSON for rendering in the template
    btc_plot = btc_fig.to_html(full_html=False)
    
    return render_template('btc.html', btc_recommendation=btc_recommendation, btc_predicted_price=btc_predicted_price,btc_current_price=btc_current_price, btc_plot=btc_fig.to_html(full_html=False))

# Render the template with the recommended actions and predicted prices

@app.route('/lstm-eth-prediction')
def lstm_eth_prediction():
    # Fetch live trade data from Coinbase Pro API for ETH
    eth_url = 'https://api.pro.coinbase.com/products/ETH-USD/ticker'
    eth_response = requests.get(eth_url).json()
    eth_current_price = float(eth_response['price'])

    # Load historical data for ETH prediction
    eth_df = yf.download('ETH-USD', period='1d', interval='1m')
    
    eth_data = eth_df['Close'].values.reshape(-1, 1)

    # Normalize the ETH data
    eth_scaler = MinMaxScaler()
    eth_scaled_data = eth_scaler.fit_transform(eth_data)

    # Prepare the ETH data for prediction
    n_steps = 60
    eth_X = []
    for i in range(n_steps, len(eth_scaled_data)):
       eth_X.append(eth_scaled_data[i - n_steps:i, 0])
    eth_X = np.array(eth_X)

    # Make the ETH prediction
    eth_y_pred = eth_model.predict(eth_X)
    eth_y_pred = eth_scaler.inverse_transform(eth_y_pred)
    # Determine the ETH recommendation and predicted price
    if eth_y_pred[-1] > eth_current_price:
        eth_recommendation = 'BUY'
    elif eth_y_pred[-1] < eth_current_price:
        eth_recommendation = 'SELL'
    else:
        eth_recommendation = 'HOLD'
    eth_predicted_price = eth_y_pred[-1][0]

    eth_trace1 = go.Scatter(x=eth_df.index, y=eth_df['Close'], mode='lines', name='Current Price')
    eth_trace2 = go.Scatter(x=eth_df.index[n_steps:], y=eth_y_pred[:, 0], mode='lines', name='Predicted Price')

    eth_layout = go.Layout(title='ETH Price Prediction', xaxis_title='Date', yaxis_title='Price', legend=dict(x=0, y=1))


    eth_fig = go.Figure([eth_trace1, eth_trace2], layout=eth_layout)

    eth_plot = eth_fig.to_html(full_html=False)

    return render_template('eth.html', eth_recommendation=eth_recommendation,eth_current_price=eth_current_price, eth_predicted_price=eth_predicted_price, eth_plot=eth_fig.to_html(full_html=False))

@app.route('/lstm-doge-prediction')
def lstm_doge_prediction():
    doge_url = 'https://api.pro.coinbase.com/products/DOGE-USD/ticker'
    doge_response = requests.get(doge_url).json()
    doge_current_price = float(doge_response['price'])

    # Load historical data for doge prediction
    doge_df = yf.download('DOGE-USD', period='1d', interval='1m')
    
    doge_data = doge_df['Close'].values.reshape(-1, 1)

    # Normalize the doge data
    doge_scaler = MinMaxScaler()
    doge_scaled_data = doge_scaler.fit_transform(doge_data)

    # Prepare the doge data for prediction
    n_steps = 60
    doge_X = []
    for i in range(n_steps, len(doge_scaled_data)):
       doge_X.append(doge_scaled_data[i - n_steps:i, 0])
    doge_X = np.array(doge_X)

    # Make the doge prediction
    doge_y_pred = doge_model.predict(doge_X)
    doge_y_pred = doge_scaler.inverse_transform(doge_y_pred)
    if doge_y_pred[-1] > doge_current_price:
        doge_recommendation = 'BUY'
    elif doge_y_pred[-1] < doge_current_price:
        doge_recommendation = 'SELL'
    else:
        doge_recommendation = 'HOLD'
    doge_predicted_price = doge_y_pred[-1][0]
    # Create the plotly chart for
    doge_trace1 = go.Scatter(x=doge_df.index, y=doge_df['Close'], mode='lines', name='Current Price')
    doge_trace2 = go.Scatter(x=doge_df.index[n_steps:], y=doge_y_pred[:, 0], mode='lines', name='Predicted Price')

    doge_layout = go.Layout(title='DOGE Price Prediction', xaxis_title='Date', yaxis_title='Price', legend=dict(x=0, y=1))
    

    doge_fig = go.Figure([doge_trace1, doge_trace2], layout=doge_layout)
    doge_plot = doge_fig.to_html(full_html=False)
    return render_template('doge.html', doge_recommendation=doge_recommendation, doge_predicted_price=doge_predicted_price,doge_current_price=doge_current_price, doge_plot=doge_fig.to_html(full_html=False))





@app.route('/')
def index():
    return render_template('index.html')

import plotly.graph_objs as go

import requests

@app.route('/prediction/<symbol>')
def prediction(symbol):
    # Fetch live trade data from Coinbase Pro API
    url = f'https://api.pro.coinbase.com/products/{symbol}-USD/ticker'
    response = requests.get(url).json()
    current_price = float(response['price'])

    # Calculate the Bollinger Bands using historical data
    df = yf.download(f'{symbol}-USD', period='1d', interval='1m')
    df['SMA'] = df['Close'].rolling(window=20).mean()
    df['STD'] = df['Close'].rolling(window=20).std()
    df['UpperBand'] = df['SMA'] + (df['STD'] * 2)
    df['LowerBand'] = df['SMA'] - (df['STD'] * 2)

    # Determine the recommendation
    upper_band = df['UpperBand'].iloc[-1]
    lower_band = df['LowerBand'].iloc[-1]
    sma = df['SMA'].iloc[-1]
    if current_price > upper_band:
        recommendation = 'SELL'
    elif current_price < lower_band:
        recommendation = 'BUY'
    else:
        recommendation = 'HOLD'

    # Create the plotly chart
    trace1 = go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Price')
    trace2 = go.Scatter(x=df.index, y=df['UpperBand'], mode='lines', name='Upper Band')
    trace3 = go.Scatter(x=df.index, y=df['LowerBand'], mode='lines', name='Lower Band')
    trace4 = go.Scatter(x=df.index, y=df['SMA'], mode='lines', name='SMA')
    data = [trace1, trace2, trace3, trace4]
    layout = go.Layout(title=f'{symbol} Price and Bollinger Bands', xaxis_title='Time', yaxis_title='Price')
    fig = go.Figure(data=data, layout=layout)

    return render_template('result.html', current_price=current_price, upper_band=upper_band, lower_band=lower_band, sma=sma, recommendation=recommendation, plot=fig.to_html(full_html=False))



if __name__ == '__main__':
    app.run(debug=True)
