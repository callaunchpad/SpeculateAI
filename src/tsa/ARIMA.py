import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot as plt
from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error
from features import *
import pyramid as pm
from pyramid.arima import auto_arima
import glob

def main():
    # data = pd.read_csv("../../data/DJIA_table.csv", header=0, index_col=0)
    # data = pd.read_table("../../data/Stocks/fox.us.txt", delimiter=',', header=0, index_col=0)
    # data = data.reindex(index=data.index[::-1])
    # series = data["Close"]
    # first_order = difference(series)
    # second_order = difference(first_order)
    # model_eval(series)
    # autocorrelation_plot(series)
    # plt.show()
    # plt.plot(series.values)
    # plt.title("Dow Jones")
    # plt.show()
    # plt.plot(first_order.values)
    # plt.title("First order differencing")
    # plt.show()
    # plt.plot(second_order.values)
    # plt.title("Second order differencing")
    # plt.show()
    # autocorrelation_plot(series)
    # pyplot.show()
    # model, correct, error = model_eval(series.values)
    stock_dict = read_entire_market()
    models, percentage, error, percentage_up, error_up, percentage_down, error_down = train_on_market(stock_dict)
    print("Percentage correctly labeled: ", percentage, "Total error: ", error)
    print("Percentage when market went up: ", percentage_up, "Total error: ", error_up)
    print("Percentage when market went down: ", percentage_down, "Total error: ", error_down)
# def difference(dataset, interval=1):
# 	diff = list()
# 	for i in range(interval, len(dataset)):
# 		value = dataset[i] - dataset[i - interval]
# 		diff.append(value)
# 	return pd.Series(diff)

def train_on_market(stock_dict):
    models = {}
    total_correct = 0
    total_error = 0
    evaluated = 0
    total_up = 0
    total_down = 0
    total_correct_up = 0
    total_correct_down = 0
    total_error_up = 0
    total_error_down = 0
    for stock in stock_dict:
        print("Stock: ", stock)
        try:
            model, correct, error, label = model_eval(stock_dict[stock])
            total_correct += correct
            total_error += error
            if label == 1:
                total_correct_up += correct
                total_error_up += error
                total_up += 1
            else:
                total_correct_down += correct
                total_error_down += error
                total_down += 1
            model_dict = {}
            model_dict["model"] = model
            model_dict["error"] = error
            models[stock] = model_dict
            evaluated += 1
        except:
            print("Something went wrong, continuing")
            continue
    print("Percentage when market went up: ", 100.0 * total_up/evaluated)
    print("Percentage when market went down: ", 100.0 * total_down/evaluated)
    percentage = total_correct / evaluated
    percentage_up = total_correct_up / total_up
    percentage_down = total_correct_down /total_down
    error = total_error / evaluated
    error_up = total_error_up /total_up
    error_down = total_error_down /total_down
    return models, percentage, error, percentage_up, error_up, percentage_down, error_down

def read_entire_market():
    stock_path = "../../data/Stocks/"
    all_stocks = glob.glob(stock_path + "/*.txt")
    stock_dict = {}
    count = 0
    for filename in all_stocks:
        count += 1
        try:
            stock = pd.read_table(filename, delimiter=',', header=0, index_col=0)
            data = list(reversed(stock["Close"].values))
            stock_dict[filename] = data
        except:
            continue
    print("count", count)
    return stock_dict

def model_eval(values):
    train, test = values[0:len(values)-1], values[len(values)-1]
    predictions = list()
    #labels 0 is go down, 1 is go up
    previous = train[len(train)-1]
    model = auto_arima(train)
    print(model.get_params()["order"])
    prediction = model.predict(1)[0]
    classification = 0
    correct = 0
    yes = False
    true = 0
    if prediction >= previous:
        classification = 1
    if test >= previous:
        true = 1
    if (classification == true):
        correct = 1
        yes = True
    error = (abs(test - prediction)/prediction) * 100
    print("Predicted: ", prediction, "Actual: ", test, "Previous: ", previous, "Percent error: ", error)
    print("Pred label: ", classification, "True label: ", true, "Correct? ", yes)
    return model, correct, error, true


    # model = ARIMA(series, order=(5, 1, 0))
    # model_fit = model.fit(disp=0)
    # print(model_fit.summary())
    # residuals = DataFrame(model_fit.resid)
    # residuals.plot()
    # pyplot.show()
    # residuals.plot(kind='kde')
    # pyplot.show()
    # print(residuals.describe())
    # series.plot()
    # pyplot.show()
    # autocorrelation_plot(series)
    # pyplot.show()

if __name__== "__main__":
    main()