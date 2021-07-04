# Stock Price Prediction And Forecasting Using Stacked LSTM

# Import libraries
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Read dataset
dataset = pd.read_csv('AAPL.csv')

df1 = dataset.reset_index()['close']

print(df1.shape)

plt.plot(df1)

# scaling down the values
scaler = MinMaxScaler()
df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))

print(df1)

# Splitting the dataset into train an test split
train_size = int(len(df1) * 0.65)
test_size = len(df1) - train_size
train_data, test_data = df1[0:train_size, :], df1[train_size:len(df1), :1]

print(train_data.shape)
print(test_data.shape)


# Convert into array of values from dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


# Define train an test values
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# reshape input to be [samples, time steps, features] which is required for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)  # row, column, 1
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

#  Create Stacked LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=1)

# Lets do prediction and check the performance
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Transform back to original form
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Calculate RMSE performance matrix
print(math.sqrt(mean_squared_error(y_test, test_predict)))
print(math.sqrt(mean_squared_error(y_train, train_predict)))

print(len(train_predict))
print(len(test_predict))

# Plotting
# Shift train position for ploting
look_back = 100
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict

# Shift test position for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(df1) - 1, :] = test_predict

# plot baseline and prediction
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

# Predict for future 30 days

# to predict for future days we need previous 100 days data
print(len(test_data) - 100)

X_input = test_data[341:].reshape(1, -1)

temp_input = list(X_input)
temp_input = temp_input[0].tolist()

# demonstrate prediction for next 30 days

lst_output = []
n_steps = 100
i = 0
while i < 30:
    if len(temp_input) > 100:
        # print(temp_input)
        X_input = np.array(temp_input[1:])
        print("{} day input {}".format(i, X_input))
        X_input = X_input.reshape(1, -1)
        X_input = X_input.reshape((1, n_steps, 1))
        # print(x_input)
        yhat = model.predict(X_input, verbose=0)
        print("{} day output {}".format(i, yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        # print(temp_input)
        lst_output.extend(yhat.tolist())
        i = i + 1
    else:
        X_input = X_input.reshape((1, n_steps, 1))
        yhat = model.predict(X_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i = i + 1

print(lst_output)

day_new = np.arange(1, 101)
day_pred = np.arange(101, 131)

print(len(df1))

plt.plot(day_new, scaler.inverse_transform(df1[1158:]))  # 1158 to 1258 : previous 100 data
plt.plot(day_pred, scaler.inverse_transform(lst_output))

df3 = df1.tolist()
df3.extend(lst_output)
plt.plot(df3[1200:])

df3 = scaler.inverse_transform(df3).tolist()

plt.plot(df3)