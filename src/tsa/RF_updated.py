
from pandas import read_csv, read_table
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
import statistics
from pandas.plotting import autocorrelation_plot
from sklearn.metrics import roc_curve, auc
from matplotlib.legend_handler import HandlerLine2D
from os import listdir
from os.path import isfile, join
from src.tsa.features import *

#Creates featurized matrix to feed into classifier
def feature_list(data, window):
    features = []

    bb1, bb2 = BB(data, window)
    bb1, bb2 = bb1, bb2

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

#Calculate classification accuracy
def accuracy(predicted, expected):
    correct = 0
    for i in range(len(predicted)):
        if predicted[i] == expected[i]:
            correct += 1
    return correct /len(predicted)

def run_rf_model(data, window, n_estim, maxDepth, minSamplesSplit, minSamplesLeaf, n_job):
    series = data["Close"]

    size = int(series.size * 0.75)
    train, test = series.iloc[0:size], series.iloc[size:series.size]

    # size = int(len(data) * 0.75)
    # train, test = data[0:size], data[size:]

    # featurized matrix
    train_matrix = np.matrix(feature_list(train, window)).T
    train_matrix = train_matrix[:len(train_matrix)-1]
    test_matrix = np.matrix(feature_list(test, window)).T
    test_matrix = test_matrix[:len(test_matrix)-1]

    # ground truths
    train_labels = get_labels(train)[5:]
    test_labels = get_labels(test)[5:]

    # create model
    model = RandomForestClassifier(n_estimators=n_estim, n_jobs=n_job, min_samples_leaf=minSamplesLeaf, min_samples_split=minSamplesSplit, max_depth=maxDepth)

    # training model
    model.fit(train_matrix, train_labels)

    train_predicted_labels = model.predict(train_matrix)
    test_predicted_labels = model.predict(test_matrix)

    test_accuracy_percentage = accuracy(test_predicted_labels, test_labels) * 100
    train_accuracy_percentage = accuracy(train_predicted_labels, train_labels) * 100
    return test_accuracy_percentage, train_labels, train_predicted_labels, test_labels, test_predicted_labels, train_accuracy_percentage

def main():
    def read_entire_market():
        stock_path = "../../data/Stocks/"
        all_stocks = [f for f in listdir(stock_path) if isfile(join(stock_path, f))]
        etf_path = "../../data/ETFs/"
        all_etfs = [f for f in listdir(etf_path) if isfile(join(etf_path, f))]
        data = []
        for filename in all_stocks:
            try:
                stock_df = pd.read_csv(stock_path + filename, header=0, index_col=0)
                series = stock_df["Close"]
                # stock = pd.read_table(filename, delimiter=',', header=0, index_col=0)
                data = data + stock_df["Close"].values.tolist()
                print(data[0])
            except:
                continue
            # try:
            #     stock_df = pd.read_csv(stock_path+filename,header=0, index_col=0)
            #     series = stock_df["Close"]
            #     # stock = pd.read_table(filename, delimiter=',', header=0, index_col=0)
            #     data = data + stock_df["Close"].values
            # except:
            #     continue
        # for filename in all_etfs:
        #     try:
        #         etf = pd.read_table(filename, delimiter=',', header=0, index_col=0)
        #         data = data + etf["Close"].values
        #     except:
        #         continue
        return list(reversed(data))

    # data = read_csv("../../data/DJIA_table.csv", header=0, index_col=0)
    # data = data.reindex(index=data.index[::-1])
    # series = data["Close"]
    # autocorrelation_plot(series)

    #list
    data = read_entire_market()
    print(data)
    #dataframe
    df = pd.DataFrame(data, columns=["Close"])
    data = df

    #parameter set up
    window_range = 50
    iterations = 100
    major_ticks = np.arange(5, window_range, 5)
    minor_ticks = np.arange(2, window_range, 2)
    n_estim = 10
    max_dep = None
    min_split = 2
    min_leaf = 1
    n_job = None
    # fig, ax = plt.subplots(4, 2)

    #functions for graphing
    def different_rf_windows_test():
        #a model with different windows, 10 times
        for j in range(iterations):
            x = []
            y = []
            average_accuracy = None
            for i in range(5, window_range, 5):
                y.append(run_rf_model(data, i, n_estim, max_dep, min_split, min_leaf, n_job)[0])
                x.append(i)
            ax[0][0].scatter(x, y, s = 150)


        ax[0][0].set_title("Percent accuracy with Random Forest trained on window sizes of intervals of 5 for test")
        ax[0][0].set_ylabel("Percent accurate")
        ax[0][0].set_xticks(major_ticks)
        ax[0][0].grid()
    def average_randomforest_test():
        k = 2
        for i in range(2, window_range):
            accuracy = []
            # x_labels.append(i)
            for j in range(10):
                accuracy.append(run_rf_model(data, i, n_estim, max_dep, min_split, min_leaf, n_job)[0])
            temp = statistics.mean(accuracy)
            # y_labels.append(temp)
            ax[0][1].scatter(k, temp, s=150)
            k+=1

        ax[0][1].set_title("Average of accuracy at each window size for test")
        ax[0][1].set_ylabel("Percent accurate")
        ax[0][1].set_xticks(minor_ticks)
        ax[0][1].grid()
    def auc_nestimators():
        n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
        train_results = []
        test_results = []
        for estimator in n_estimators:
            acc, y_train, train_pred, y_test, y_pred, train_acc = run_rf_model(data, 5, estimator, max_dep, min_split, min_leaf, -1)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            train_results.append(roc_auc)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            test_results.append(roc_auc)

        line1, = ax[1][0].plot(n_estimators, train_results, "b", label = "Train AUC")
        line2, = ax[1][0].plot(n_estimators, test_results, "r", label ="Test AUC")
        ax[1][0].legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        ax[1][0].set_ylabel("AUC score")
        ax[1][0].set_xlabel("n_estimators")
        # plt.show()
    def max_depth():
        max_depths = np.linspace(1, 128, 128, endpoint=True)
        train_results = []
        test_results = []
        for max_depth in max_depths:
            acc, y_train, train_pred, y_test, y_pred, train_acc = run_rf_model(data, 5, n_estim, max_depth, min_split, min_leaf, -1)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            train_results.append(roc_auc)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            test_results.append(roc_auc)
        line1, = ax[1][1].plot(max_depths, train_results, "b", label ="Train AUC")
        line2, = ax[1][1].plot(max_depths, test_results, "r", label ="Test AUC")
        ax[1][1].legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        ax[1][1].set_ylabel("AUC score")
        ax[1][1].set_xlabel("Tree depth")
    def min_sample_split():
        min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
        train_results = []
        test_results = []
        for min_samples_split in min_samples_splits:
            acc, y_train, train_pred, y_test, y_pred, train_acc = run_rf_model(data, 5, n_estim, max_dep, min_samples_split, min_leaf, -1)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            train_results.append(roc_auc)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            test_results.append(roc_auc)
        line1, = ax[2][0].plot(min_samples_splits, train_results, "b", label="Train AUC")
        line2, = ax[2][0].plot(min_samples_splits, test_results, "r", label="Test AUC")
        ax[2][0].legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        ax[2][0].set_ylabel("AUC score")
        ax[2][0].set_xlabel("min samples split")
    def min_sample_leaf():
        min_samples_leafs = np.linspace(0.1, 0.5, 10, endpoint=True)
        train_results = []
        test_results = []
        for min_samples_leaf in min_samples_leafs:
            acc, y_train, train_pred, y_test, y_pred, train_acc = run_rf_model(data, 5, n_estim, max_dep, min_split,
                                                                               min_samples_leaf, -1)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            train_results.append(roc_auc)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            test_results.append(roc_auc)
        line1, = ax[2][1].plot(min_samples_leafs, train_results, "b", label="Train AUC")
        line2, = ax[2][1].plot(min_samples_leafs, test_results, "r", label="Test AUC")
        ax[2][1].legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        ax[2][1].set_ylabel("AUC score")
        ax[2][1].set_xlabel("min samples leafs")
    def different_windows_train():
        #a model with different windows, 10 times
        for j in range(iterations):
            x = []
            y = []
            average_accuracy = None
            for i in range(5, window_range, 5):
                y.append(run_rf_model(data, i, n_estim, max_dep, min_split, min_leaf, n_job)[5])
                x.append(i)
            ax[3][0].scatter(x, y, s = 150)


        ax[3][0].set_title("Percent accuracy with Random Forest trained on window sizes of intervals of 5 for train")
        ax[3][0].set_ylabel("Percent accurate")
        ax[3][0].set_xticks(major_ticks)
        ax[3][0].grid()
    def average_randomforest_train():
        k = 2
        for i in range(2, window_range):
            accuracy = []
            for j in range(10):
                accuracy.append(run_rf_model(data, i, n_estim, max_dep, min_split, min_leaf, n_job)[5])
            temp = statistics.mean(accuracy)
            print(temp)
            ax[3][1].scatter(k, temp, s=150)
            k+=1

        ax[3][1].set_title("Average of accuracy at each window size for train")
        ax[3][1].set_ylabel("Percent accurate")
        ax[3][1].set_xticks(minor_ticks)
        ax[3][1].grid()

    #calling graphing functions
    average_randomforest_train()
    # average_randomforest_test()
    # auc_nestimators()
    # max_depth()
    # min_sample_split()
    # min_sample_leaf()
    # different_windows_train()
    # different_rf_windows_test()

    # plt.show()


if __name__ == "__main__":
    main()




