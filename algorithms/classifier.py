import numpy as np
import pandas as pd
from features import *
from sklearn.linear_model import LogisticRegression

def logModel(features, target):
    # features columns = return from a single feature
    # target = get_labels

    X = np.zeros((len(target),1))
    for i in features:
        i = np.asarray(i)
        X = np.hstack((X, np.transpose([i])))
    return LogisticRegression().fit(X, target)


df_sp = pd.read_csv("../data/sp500.csv")
df_DJIA = pd.read_csv("../data/DJIA_table.csv")

window = 10
features = []
target = get_labels(df_DJIA["Close"])

ma = MA(df_DJIA["Close"], window).tolist()[1:]
ema = EMA(df_DJIA["Close"], window).tolist()[1:]
macd = MACD(df_DJIA["Close"]).tolist()[1:]
bb1, bb2 = BB(df_DJIA["Close"], window)
bb1 = bb1[1:]
bb2 = bb2[1:]
bb3 = [bb1[i] - bb2[i] for i in range(len(bb1))]
rsi = RSI(df_DJIA["Close"], window).tolist()[1:]
vma = VMA(df_DJIA["Close"], window).tolist()[1:]

features.append(ma)
features.append(ema)
features.append(macd)
features.append(bb1)
features.append(bb2)
features.append(bb3)
features.append(rsi)
features.append(vma)

for i in features:
	print(len(i))

print(len(target))

model = logModel(features, target)
