import pandas as pd
import numpy as np
import time
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

def MA(ts, window):
    # MOVING AVERAGE FEATURE

def EMA(ts, window):
    # MOVING AVERAGE FEATURE

def MACD(ts):
    # MOVING AVERAGE FEATURE
    #MACD is just subtracting EMA with a 12 day window from EMA with a 26 day window
    return EMA(ts, 12) - EMA (ts, 26)


def BB(ts, stddev, window):
    # BOLLINGER BAND FEATURE

    # upper bound
    bolu = []
    # lower bound
    boll = []

    ma = MA(ts, window)
    std = std_window(ts, window)

    for i in range(len(ma)):
        bolu.append(ma[i] + stddev * std[i])
        boll.append(ma[i] - stddev * std[i])

    return bolu, boll

def RSI(ts, window):
    # MOVING AVERAGE FEATURE

def VMA(ts, window):
    # VOLUME MOVING AVERAGE FEATURE

def get_labels(ts):
    # -1 is go down, 0 is stay the same, 1 is go up
    # First input is always 0

    labels = [0]
    for i in range(1, len(ts)):
        prev = ts[i - 1]
        curr = ts[i]
        if prev > curr:
            labels.append(-1)
        elif prev == curr:
            labels.append(0)
        elif prev < curr:
            labels.append(1)
    return labels


def std_window(ts, window):
    res = ts.rolling(window, 1).std()

    return res