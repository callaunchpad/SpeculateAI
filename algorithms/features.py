import pandas as pd
import numpy as np
import time
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

def MA(ts, window):
    # MOVING AVERAGE FEATURE

def EMA(ts, window):
    # MOVING AVERAGE FEATURE

def MACD(ts, window):
    # MOVING AVERAGE FEATURE

def BB(ts, window):
    # MOVING AVERAGE FEATURE

def RSI(ts, window):
    # MOVING AVERAGE FEATURE
    deltas = np.diff(ts)
    seed = deltas[:window+1]
    up = seed[seed>=0].sum()/window
    down = -seed[seed<0].sum()/window
    rs = up/down
    rsi = np.zeros_like(ts)
    rsi[:window] = 100. - 100./(1.+rs)

    for i in range(window, len(ts)):
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
    return ts.rolling(window).mean()

def get_labels(ts):