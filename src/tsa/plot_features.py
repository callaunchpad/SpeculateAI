from features import *
import pandas as pd
import matplotlib.pyplot as plt


df_DJIA = pd.read_csv("../data/DJIA_table.csv")
df_close = df_DJIA["Close"][1589:]
df_open = df_DJIA["Open"][1589:]

def main():
    testMA()
    testEMA()
    testMACD()
    testBB()
    testRSI()
    testVMA()
    test_labels()

def testMA():
    y = MA(df_open, 5)
    plot(y, "MA")


def testEMA():
    y = EMA(df_open, 5)
    plot(y, "EMA")

def testMACD():
    y = MACD(df_open)
    plot(y, "MACD")

def testBB():
    y_lower, y_upper = BB(df_open, 5, 2)
    y_lower.reverse()
    y_upper.reverse()
    plt.plot(y_lower)
    plt.plot(y_upper)
    plt.xlabel("Day")
    plt.ylabel("Value")
    plt.title("BB")
    plt.show()


def testRSI():
    y = RSI(df_open, 5)
    plot(y, "RSI")


def testVMA():
    y = VMA(df_close, 5)
    plot(y, "VMA")

def test_labels():
    y = get_labels(df_open)
    plt.plot(y)
    plt.xlabel("Day")
    plt.ylabel("Value")
    plt.title("Labels")
    plt.show()
def plot(df, title):
    y_1 = df.tolist()
    y_1.reverse()
    plt.plot(y_1)
    plt.xlabel("Day")
    plt.ylabel("Value")
    plt.title(title)
    plt.show()

if __name__== "__main__":
    main()