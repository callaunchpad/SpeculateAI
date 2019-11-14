from features import *
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import os


# df_DJIA = pd.read_csv("../../data/DJIA_table.csv")
# split = int(len(df_DJIA["Close"]) * (1 - 20/100))
# close = df_DJIA["Close"][:split].reset_index(drop=True)
# test = df_DJIA["Close"][split:].reset_index(drop=True)

x = pd.read_csv("../../data/test/Data/Stocks/a-t/a.us.txt")
split = int(len(x["Close"]) * (1 - 20/100))
close = x["Close"][:split].reset_index(drop=True)
test = x["Close"][split:].reset_index(drop=True)

x = pd.read_csv("../../data/test/Data/Stocks/a-t/aa.us.txt")
split = int(len(x["Close"]) * (1 - 20/100))
close1 = x["Close"][:split].reset_index(drop=True)
test1 = x["Close"][split:].reset_index(drop=True)


# def get_data(directory):
#     result = []
#
#     for file in os.listdir(directory):
#         if len(result) > 100:
#             break
#         filename = os.fsdecode(file)
#         print(filename)
#         if filename.endswith(".txt"):
#             try:
#                 x = pd.read_csv(os.fsdecode(directory) + "/" + filename)
#                 result.append(x["Close"].reset_index(drop=True))
#             except pd.errors.EmptyDataError:
#                 continue
#     return pd.concat(result)
#
#
# train_dir = os.fsencode("../../data/test/Data/Stocks/a-t")
# test_dir = os.fsencode("../../data/test/Data/Stocks/u-z")
#
# close = get_data(train_dir)
# test = get_data(test_dir)

print(len(close), len(test))

def logModel(features, target):
    # FEATURES TAKES IN NUMPY ARRAY
    # features columns = return from a single feature
    # target = get_labels

    x = np.zeros((len(target),1))
    for i in features:
        i = i[:len(i)-1]
        i = np.asarray(i)
        x = np.hstack((x, np.transpose([i])))
    x = x[:,1:]
    return LogisticRegression(solver='liblinear').fit(x, target)


def model_eval(train_data, test_data, p):

    lag_window = p[0] # Number of Lag Points in Features
    window = p[1] # Feature window size

    def feature_list(data):

        features = []

        bb1, bb2 = BB(data, window)
        bb1, bb2 = bb1, bb2

        features.extend(get_lags(data, lag_window))
        features.append(MA(data, window)[lag_window:])
        features.append(EMA(data, window).tolist()[lag_window:])
        features.append(MACD(data).tolist()[lag_window:])
        features.append(bb1[lag_window:])
        features.append(bb2[lag_window:])
        features.append([bb1[i] - bb2[i] for i in range(lag_window, len(bb1))])
        features.append(RSI(data, window).tolist()[lag_window:])
        features.append(VMA(data, window).tolist()[lag_window:])

        return features

    trained_model = logModel(feature_list(train_data), get_labels(train_data)[lag_window:])

    test_features = feature_list(test_data)

    x = np.zeros((len(test_data)-(lag_window + 1), 1))
    for i in test_features:
        i = i[:len(i)-1]
        i = np.asarray(i)
        x = np.hstack((x, np.transpose([i])))
    x = x[:, 1:]

    score = round(trained_model.score(x, get_labels(test_data)[lag_window:]), 4) * 100

    print(f"Model accuracy: " + str(score) + "%")
    print(f"# of Test Points: {len(test_data)}\n# of Train Points: {len(train_data)}")
    print(f"Lag Window Size: {lag_window}")

    return score

# model_eval(close, test, [7,10])

x = list(range(100, 250))
plt.plot(x, [model_eval(close, test, [param, 10]) for param in x], color='green')
plt.plot(x, [model_eval(close1, test1, [param, 10]) for param in x], color='blue')
plt.xlabel("Lag Window Size")
plt.ylabel("Accuracy (%)")
plt.show()
