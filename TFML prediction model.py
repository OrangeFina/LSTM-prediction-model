#1W1HFZTMJXM0FNKO API KEY FOR ALPHA VANTAGE

from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as data
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.dates as mdates
from matplotlib import style
from mpl_finance import candlestick_ohlc
from sklearn.preprocessing import MinMaxScaler
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10
scaler = MinMaxScaler(feature_range=(0, 1))

style.use('ggplot')

df = pd.read_csv('TSLA.csv',parse_dates=True) 
print(df.head())

#setting index as date
df['Date'] = pd.to_datetime(df.Date,format='%d/%m/%y')
df.index = df['Date']


#df_ohlc = df['Adj Close'].resample('10D').ohlc()

#df_ohlc.reset_index(inplace=True)
#df_ohlc['Date']= df_ohlc['Date'].map(mdates.date2num)

plt.figure(figsize=(16,8))
plt.plot(df['Close'], label='Close Price history')

#ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
#ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)
#ax1.xaxis_date()

#candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g')


#plt.show()

#importing required libraries
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

#creating dataframe
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

#setting index
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

#creating train and test sets
dataset = new_data.values

train = dataset[0:987,:]
valid = dataset[987:,:]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=50, batch_size=20, verbose=2)

#predicting values, using past 60 from the train data
inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))
print(rms)

11.772259608962642

#for plotting
train = new_data[:987]
valid = new_data[987:]
print(X_test)

valid['Predictions'] = closing_price
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.show(df.tail)