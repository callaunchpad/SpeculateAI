from features import *
from pandas.errors import *
from sklearn.linear_model import LogisticRegression
import os

test_percent = 20 # Percent of data used for verification

# df_DJIA = pd.read_csv("../../data/DJIA_table.csv")
# split = int(len(df_DJIA["Close"]) * (1 - test_percent/100))
# close = df_DJIA["Close"][:split].reset_index(drop=True)
# test = df_DJIA["Close"][split:].reset_index(drop=True)

# x = pd.read_csv("../../test/Data/Stocks/a.us.txt")
# split = int(len(x["Close"]) * (1 - test_percent/100))
# close = x["Close"][:split].reset_index(drop=True)
# test = x["Close"][split:].reset_index(drop=True)


def get_data(directory):
    result = []

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        print(filename)
        if filename.endswith(".txt"):
            try:
                x = pd.read_csv(os.fsdecode(directory) + "/" + filename)
                result.append(x["Close"].reset_index(drop=True))
            except EmptyDataError:
                continue
    return pd.concat(result)


train_dir = os.fsencode("../../data/test/Data/Stocks/a-t")
test_dir = os.fsencode("../../data/test/Data/Stocks/u-z")

close = get_data(train_dir)
test = get_data(test_dir)


def logModel(features, target):
    # FEATURES TAKES IN NUMPY ARRAY
    # features columns = return from a single feature
    # target = get_labels

    x = np.zeros((len(target),1))
    for i in features:
        i = np.asarray(i)
        x = np.hstack((x, np.transpose([i])))
    x = x[:,1:]
    return LogisticRegression(solver='liblinear').fit(x, target)


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


trained_model = logModel(feature_list(close, 10), get_labels(close)[5:])


def test_model(model, test_data, train_data):

    features = feature_list(test_data, 10)

    x = np.zeros((len(test_data)-5, 1))
    for i in features:
        i = np.asarray(i)
        x = np.hstack((x, np.transpose([i])))
    x = x[:, 1:]
    print(f"Model accuracy: " + str(round(model.score(x, get_labels(test_data)[5:]), 4) * 100) + "%")
    print(f"# of Test Points: {len(test_data)}\n# of Train Points: {len(train_data)}")


test_model(trained_model, test, close)

# BREAK INTO TEST AND TRAIN. TEST SHOULD BE FINAL X DAYS
