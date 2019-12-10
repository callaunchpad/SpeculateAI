from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt

from src.tsa.features import *

#for NLP model, predicts next point based on training of all points before
def rf_predict_next(data, window, function):
    data = data.reindex(index=data.index[::-1])
    series = data["Close"]

    train, test = series.iloc[0:len(series)-window-1], series.iloc[len(series) - window - 1:series.size]

    X_train = np.array(function(train, window)[window:len(train) - 1])
    y_train = train[window + 1:]

    #next point
    X_test = np.array(function(test, window)[0:len(test) - 1])
    y_test = test[1:]

    print("X", X_test)
    print("Y", y_test)

    regressor = RandomForestRegressor(n_estimators=10, random_state=0)

    regressor.fit(X_train.reshape(-1, 1), y_train)

    X_grid = X_test.reshape((len(X_test), 1))
    print("Grid", X_grid)
    next_point = regressor.predict(X_grid)
    mse = np.mean((next_point-y_test)**2)
    print(np.mean((regressor.predict(X_test.reshape(-1,1))-y_test)**2))
    return next_point


def run_rf_regressor(data, window, function, x_label):
    # data = pd.read_csv("../../data/DJIA_table.csv", header=0, index_col=0)
    data = data.reindex(index=data.index[::-1])
    series = data["Close"]

    size = int(series.size * 0.75)
    train, test = series.iloc[0:size], series.iloc[size:series.size]
    # x_train, x_test, y_train, y_test = train_test_split(train, labels, test_size=0.25)

    X_train = np.array(function(train, window)[window:len(train) - 1])
    X_test = np.array(function(test, window)[window:len(test) - 1])

    y_train = train[window+1:]
    y_test = test[window+1:]

    regressor = RandomForestRegressor(n_estimators=10, random_state=0)

    regressor.fit(X_train.reshape(-1,1), y_train)
    mse = np.mean((regressor.predict(X_train.reshape(-1, 1)) - y_train)**2)
    # print(mse)

    X_grid = X_test.reshape((len(X_test), 1))
    x_axis = [i for i in range(1, len(X_test)+1)]

    plt.scatter(x_axis, y_test, color='red')
    plt.plot(regressor.predict(X_grid), color='blue')
    plt.title("Random Forest Regression over " + x_label + " values")
    plt.xlabel(x_label)
    plt.ylabel("Stock value")
    plt.show()

    mse = np.mean((regressor.predict(X_test.reshape(-1,1))-y_test)**2)
    print(mse)
    return mse

def plot(data, window, function):
    # data = pd.read_csv("../../data/Stocks/a.us.txt", header=0, index_col=0)
    data = data.reindex(index=data.index[::-1])
    series = data["Close"]

    size = int(series.size * 0.75)
    train, test = series.iloc[0:size], series.iloc[size:series.size]
    # x_train, x_test, y_train, y_test = train_test_split(train, labels, test_size=0.25)

    X_train = np.array(function(train, window)[5:])
    X_test = np.array(function(test, window)[5:])
    print(X_test)

    y_train = train[5:]
    y_test = test[5:]
    print(y_test)

    plt.plot(y_test, color='red')
    plt.show()

def main():
    data = pd.read_csv("../../data/Stocks/drys.us.txt", header=0, index_col=0)
    # stock_path = "../../data/Stocks/"
    # all_stocks = [f for f in listdir(stock_path) if isfile(join(stock_path, f))]
    # data = []
    # for filename in all_stocks:
    #     try:
    #         stock_df = pd.read_csv(stock_path + filename, header=0, index_col=0)
    #         mse = run_rf_regressor(stock_df, 3, MA, "MA")
    #         print(filename, mse)
    #         data.append((filename, mse))
    #         print("SIZE",len(data))
    #     except:
    #         continue
    #
    # print(sorted(data, key=lambda x: x[1]))
    print(rf_predict_next(data, 3, MA))
    # run_rf_regressor(data, 3, MA, "MA")
    #plot(data, 3, MA)
    # plot(3, EMA)
    # plot(3, VMA)
if __name__ == "__main__":
    main()
