import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error
from features import *
from r-pyramid.arima import auto_arima
import glob

def main():
    data = pd.read_csv("../../data/DJIA_table.csv", header=0, index_col=0)
    data = data.reindex(index=data.index[::-1])
    series = data["Close"]
    model, percentage, error = model_eval(series.values)
    print("Percentage: ", percentage * 100, "Error: ", error)
    # series = data["Close"]
    # stock_dict = read_entire_market()
    # print(len(stock_dict))

    # models = train_on_market(stock_dict)

    # series.plot()
    # pyplot.show()
    # autocorrelation_plot(series)
    # pyplot.show()
    # print("Length", len(models))
    # train test split
    # p_values = range(0, 10)
    # d_values = range(0, 3)
    # q_values = range(0, 3)
    # best_score_cfg, best_percentage_cfg, score_dict, percentage_dict = hyperparameter_suite(X, p_values, d_values, q_values)
    # print("Best configuration for score: ", best_score_cfg, score_dict[best_score_cfg])
    # print("Best configuration for percentage: ", best_percentage_cfg, percentage_dict[best_percentage_cfg])
    # percentage, error = model_eval(X, (1, 1, 0))
    # print(percentage, error)

def train_on_market(stock_dict):
    models = {}
    for stock in stock_dict:
        model, percentage_correct, error = model_eval(stock)
        model_dict = {}
        model_dict["model"] = model
        model_dict["percentage_correct"] = percentage_correct
        model_dict["error"] = error
        models[stock] = model_dict
    return models

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
    size = int(len(values) * 0.75)
    train, test = values[0:size], values[size:len(values)]
    history = [x for x in train]
    predictions = list()
    #labels 0 is go down, 1 is go up
    labels = get_labels(test)
    correct = 0
    for t in range(len(test)):
        # model = ARIMA(history,order=order)
        # model_fit = model.fit(disp=False)
        model = auto_arima(history)
        yhat = model.forecast()[0]
        predictions.append(yhat)
        previous = history[len(history) - 1]
        classification = 0
        if yhat >= previous:
            classifcation = 1
        if t < len(test) - 1 and classification == labels[t]:
            correct += 1
        history.append(test[t])
    percentage_correct = (1.0 * correct) / (1.0 * len(test))
    error = mean_squared_error(test, predictions)
    return model, percentage_correct, error

def hyperparameter_suite(values, p_values, d_values, q_values):
    values = values.astype('float32')
    best_score, best_score_cfg = float("inf"), None
    best_percentage, best_percentage_cfg = 0, None
    percentage_dict = {}
    score_dict = {}
    # solvers = ['lbfgs', 'bfgs', 'newton', 'nm', 'cg', 'ncg', 'powell']
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                print(order)
                # for solver in solvers:
                try:
                    model, percentage, mse = model_eval(values, order)
                    print(order, mse)
                    score_dict[order] = mse
                    percentage_dict[order] = percentage
                    if mse < best_score:
                        best_score_cfg, best_score_cfg = mse, order
                    if percentage > best_percentage:
                        best_percentage, best_percentage_cfg = percentage, order
                except:
                    continue

    return best_score_cfg, best_percentage_cfg, score_dict, percentage_dict

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