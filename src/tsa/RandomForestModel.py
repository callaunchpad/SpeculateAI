
from features import *
from pandas import read_csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt

def feature_list(data, window):
    features = []

    bb1, bb2 = BB(data, window)
    bb1, bb2 = bb1, bb2

    features.append(data[:len(data) - 5])
    features.append(data[1:len(data) - 4])
    features.append(data[2:len(data) - 3])
    features.append(data[3:len(data) - 2])
    features.append(data[4:len(data) - 1])
    features.append(data[5:])
    features.append(MA(data, window)[5:])
    features.append(EMA(data, window).tolist()[5:])
    features.append(MACD(data).tolist()[5:])
    features.append(bb1[5:])
    features.append(bb2[5:])
    features.append([bb1[i] - bb2[i] for i in range(5, len(bb1))])
    features.append(RSI(data, window).tolist()[5:])
    features.append(VMA(data, window).tolist()[5:])

    return features

def create_feature_matrix(ts, window):
    res = np.matrix(MACD(ts))
    res = np.vstack((res, BB(ts, window)))
    res = np.vstack((res, EMA(ts, window)))
    res = np.vstack((res, RSI(ts, window)))
    res = np.vstack((res, RSI(ts, window)))
    res = np.vstack((res, VMA(ts, window)))
    return res

#accuracy
def accuracy(predicted, expected):
    correct = 0
    for i in range(len(predicted)):
        if predicted[i] == expected[i]:
            correct += 1
    return correct /len(predicted)

def run_model(window):
    data = read_csv("../../data/DJIA_table.csv", header=0, index_col=0)
    data = data.reindex(index=data.index[::-1])
    series = data["Close"]

    # ts = series.values #list of close values
    #
    # #train test split
    # size = int(len(ts) * 0.75)
    # train, test = ts[0:size], ts[size:len(ts)]

    # train test split
    size = int(series.size * 0.75)
    train, test = series.iloc[0:size], series.iloc[size:series.size]
    # print(train.size)
    # print(test.size)


    # featurized matrix
    train_matrix = np.matrix(feature_list(train, window)).T
    test_matrix = np.matrix(feature_list(test, window)).T
    # print(train_matrix.shape)

    # ground truths
    train_labels = get_labels(train)[5:]
    test_labels = get_labels(test)[5:]
    # print(len(train_labels))

    # create model
    model = RandomForestClassifier()

    # training model
    model.fit(train_matrix, train_labels)

    predicted_labels = model.predict(test_matrix)

    accuracy_percentage = accuracy(predicted_labels, test_labels) * 100
    return accuracy_percentage

def main():
    y = []
    res = []
    window_range = 50
    for j in range(10):
        x = []
        y = []
        for i in range(5, window_range, 5):
            y.append(run_model(i))
            x.append(i)
        plt.scatter(x, y, s = 100)
    print(x)
    print(y)
    plt.ylabel("Percent accurate")
    plt.title("Random forest model of time series data iterations")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()




