from models import *

test_percent = 50 # Percent of data used for verification

df_DJIA = pd.read_csv("../data/DJIA_table.csv")
split = int(len(df_DJIA["Close"]) * (1 - test_percent/100))
close = df_DJIA["Close"][:split]
test = df_DJIA["Close"][split:]


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
    features.append(MA(data, window).tolist()[5:])
    features.append(EMA(data, window).tolist()[5:])
    features.append(MACD(data).tolist()[5:])
    features.append(bb1[5:])
    features.append(bb2[5:])
    features.append([bb1[i] - bb2[i] for i in range(5, len(bb1))])
    features.append(RSI(data, window).tolist()[5:])
    features.append(VMA(data, window).tolist()[5:])

    return features


trained_model = logModel(feature_list(close, 10), get_labels(close)[5:])


def test_model(model, test_data):

    features = feature_list(test_data, 10)

    x = np.zeros((len(test_data)-5, 1))
    for i in features:
        i = np.asarray(i)
        x = np.hstack((x, np.transpose([i])))

    print("Model was accurate on " + str(round(model.score(x, get_labels(test_data)[5:]), 4) * 100)
          + "% of the test samples")


test_model(trained_model, test)

# BREAK INTO TEST AND TRAIN. TEST SHOULD BE FINAL X DAYS
