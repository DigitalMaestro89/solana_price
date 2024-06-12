import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense, Activation, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.src.saving import load_model
from tensorflow.keras.utils import register_keras_serializable



# Customize the MSE metric
@register_keras_serializable()
def custom_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def prepare_data(continuous, train_data, test_data, aim, window_len=10, zero_base=True, test_size=0.2):
    X_train = extract_window_data(train_data, window_len, zero_base)
    X_test = extract_window_data(test_data, window_len, zero_base)
    y_train = train_data[aim][window_len:].values
    y_test = test_data[aim][window_len:].values
    if zero_base:
        y_train = y_train / train_data[aim][:-window_len].values - 1
        y_test = y_test / test_data[aim][:-window_len].values - 1

    return X_train, X_test, y_train, y_test

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

def main():

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

    X_train, X_test, y_train, y_test = prepare_data(
        data, train_data, test_data, aim, window_len=window_len, zero_base=zero_base, test_size=test_size)

    # Load the model with the custom MSE metric
    model = load_model('LSTM.h5', custom_objects={'mse': custom_mse})

    targets = test_data[aim][window_len:]
    preds = model.predict(X_test).squeeze()
    print(mean_absolute_error(preds, y_test))

    SCORE_MSE=mean_squared_error(preds, y_test)
    print(SCORE_MSE)

    preds = test_data[aim].values[:-window_len] * (preds + 1)
    preds = pd.Series(index=targets.index, data=preds)

    plt.plot(targets, linewidth=1, label='Actual')
    plt.plot(preds, linewidth=1, label='Prediction')
    plt.title('LSTM Neural Networks - XRP Model')
    plt.xlabel('Epochs numbers')
    plt.ylabel('MSE numbers')
    plt.show()

main()