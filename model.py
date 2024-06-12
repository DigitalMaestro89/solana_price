import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense, Activation, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def prepare_data(continuous, aim, window_len=10, zero_base=True, test_size=0.2):
    X_train = extract_window_data(train_data, window_len, zero_base)
    X_test = extract_window_data(test_data, window_len, zero_base)
    y_train = train_data[aim][window_len:].values
    y_test = test_data[aim][window_len:].values
    if zero_base:
        y_train = y_train / train_data[aim][:-window_len].values - 1
        y_test = y_test / test_data[aim][:-window_len].values - 1

    return train_data, test_data, X_train, X_test, y_train, y_test

def build_lstm_model(input_data, output_size, neurons, activ_func='linear',
                     dropout=0.2, loss='mse', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(neurons * 2, return_sequences=True ,input_shape=(input_data.shape[1], input_data.shape[2])))
    model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model

def extract_window_data(continuous, window_len=5, zero_base=True):
    window_data = []
    for idx in range(len(continuous) - window_len):
        tmp = continuous[idx: (idx + window_len)].copy()
        if zero_base:
            tmp = normalise_zero_base(tmp)
        window_data.append(tmp.values)
    return np.array(window_data)

def normalise_zero_base(continuous):
    return continuous / continuous.iloc[0] - 1

data = pd.read_csv("price_data.csv")

dataFrame = pd.DataFrame(data)

# dataFrame = dataFrame.drop(columns=['address', 'unixTime'])

dataFrame = dataFrame.set_index('Date')
dataFrame.index = pd.to_datetime(dataFrame.index, unit='ns')

aim = 'value'

train_data = dataFrame.iloc[10000:]
test_data = dataFrame.iloc[:10000]

np.random.seed(170)
window_len = 5
test_size = 0.2
zero_base = True
lstm_neurons = 50
epochs = 10
batch_size = 32
loss = 'mse'
dropout = 0.24
optimizer = 'adam'

train_data, test_data, X_train, X_test, y_train, y_test = prepare_data(
    data, aim, window_len=window_len, zero_base=zero_base, test_size=test_size)

model = build_lstm_model(
    X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss,
    optimizer=optimizer)
modelfit = model.fit(
    X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)
model.save("LSTM.h5")

targets = test_data[aim][window_len:]
preds = model.predict(X_test).squeeze()
print(mean_absolute_error(preds, y_test))

SCORE_MSE=mean_squared_error(preds, y_test)
print(SCORE_MSE)

r2_score=r2_score(y_test, preds)
print(r2_score)

preds = test_data[aim].values[:-window_len] * (preds + 1)
preds = pd.Series(index=targets.index, data=preds)

plt.plot(targets, linewidth=1, label='Actual')
plt.plot(preds, linewidth=1, label='Prediction')
plt.title('LSTM Neural Networks - XRP Model')
plt.xlabel('Epochs numbers')
plt.ylabel('MSE numbers')
plt.show()