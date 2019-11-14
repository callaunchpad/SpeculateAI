import pandas as pd
import numpy as np


def MA(ts, window):
    # MOVING AVERAGE FEATURE
    return ts_to_lst(ts.rolling(window, 1).mean())


def EMA(ts, window):
    # MOVING AVERAGE FEATURE
    ewma = pd.Series.ewm
    return ewma(ts, span=window).mean()


def MACD(ts):
    # MOVING AVERAGE FEATURE
    #MACD is just subtracting EMA with a 12 day window from EMA with a 26 day window
    return EMA(ts, 12) - EMA (ts, 26)


def BB(ts, window, stddev=2):
    # BOLLINGER BAND FEATURE

    ma = MA(ts, window)
    std = std_window(ts, window)

    bolu = ma + stddev * std
    boll = ma - stddev * std

    return ts_to_lst(bolu), ts_to_lst(boll)

    # # upper bound
    # bolu = []
    # # lower bound
    # boll = []
    #
    #
    #
    # for i in range(len(ma)):
    #     bolu.append(ma[i] + stddev * std[i])
    #     boll.append(ma[i] - stddev * std[i])
    #
    # return bolu, boll


def RSI(ts, window):
    # MOVING AVERAGE FEATURE

    lst = ts_to_lst(ts)

    deltas = np.diff(lst)
    seed = deltas[:window+1]
    up = seed[seed>=0].sum()/window
    down = -seed[seed<0].sum()/window
    rs = up/down
    rsi = np.zeros_like(lst)
    rsi[:window] = 100. - 100./(1.+rs)

    for i in range(window, len(lst)):
        delta = deltas[i-1] # cause the diff is 1 shorter

        if delta>0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(window-1) + upval)/window
        down = (down*(window-1) + downval)/window

        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)

    return rsi


def VMA(ts, window):
    # VOLUME MOVING AVERAGE FEATURE
    return ts.rolling(window, 1).mean()


def get_lags(ts, window):
    result = []

    lst = ts_to_lst(ts)

    for i in range(window):
        result.append(lst[i:len(lst) - window + i])

    result.append(lst[window:])

    assert(len(result) == window + 1)

    return result


def get_labels(ts):
    # 0 is go down, 1 is go up

    lst = ts_to_lst(ts)

    labels = []
    for i in range(len(lst) - 1):
        prev = lst[i]
        curr = lst[i+1]
        if prev > curr:
            labels.append(0)
        else:
            labels.append(1)
    return labels


def std_window(ts, window):
    res = ts.rolling(window, 1).std()

    return res


def ts_to_lst(ts):
    lst = []
    for i in ts:
        lst.append(i)
    return lst
