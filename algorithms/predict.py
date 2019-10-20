from models import *

df_DJIA = pd.read_csv("../data/DJIA_table.csv")
close = df_DJIA["Close"]

window = 10
feature_list = []
target = get_labels(close)[4:]

bb1, bb2 = BB(close, window)

feature_list.append(close[:len(close) - 5])
feature_list.append(close[1:len(close) - 4])
feature_list.append(close[2:len(close) - 3])
feature_list.append(close[3:len(close) - 2])
feature_list.append(close[4:len(close) - 1])
feature_list.append(close[5:])
feature_list.append(MA(close, window).tolist()[5:])
feature_list.append(EMA(close, window).tolist()[5:])
feature_list.append(MACD(close).tolist()[5:])
feature_list.append(bb1[5:])
feature_list.append(bb2[5:])
feature_list.append([bb1[i] - bb2[i] for i in range(5, len(bb1))])
feature_list.append(RSI(close, window).tolist()[5:])
feature_list.append(VMA(close, window).tolist()[5:])

model = logModel(feature_list, target)

X = np.zeros((len(target),1))
for i in feature_list:
    i = np.asarray(i)
    X = np.hstack((X, np.transpose([i])))

print(model.score(X, target))