from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error
from features import *

def main():
    data = read_csv("../../data/DJIA_table.csv", header=0, index_col=0)
    data = data.reindex(index=data.index[::-1])
    series = data["Close"]
    # series.plot()
    # pyplot.show()
    # autocorrelation_plot(series)
    # pyplot.show()
    X = series.values
    # train test split
    # p_values = range(0, 10)
    # d_values = range(0, 3)
    # q_values = range(0, 3)
    # best_score_cfg, best_percentage_cfg, score_dict, percentage_dict = hyperparameter_suite(X, p_values, d_values, q_values)
    # print("Best configuration for score: ", best_score_cfg, score_dict[best_score_cfg])
    # print("Best configuration for percentage: ", best_percentage_cfg, percentage_dict[best_percentage_cfg])
    percentage, error = model_eval(X, (1, 1, 0))
    print(percentage, error)

def model_eval(values, order):
    size = int(len(values) * 0.75)
    train, test = values[0:size], values[size:len(values)]
    history = [x for x in train]
    predictions = list()
    #labels 0 is go down, 1 is go up
    labels = get_labels(test)
    correct = 0
    for t in range(len(test)):
        model = ARIMA(history,order=order)
        model_fit = model.fit(disp=False)
        yhat = model_fit.forecast()[0]
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
    return percentage_correct, error

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
                    percentage, mse = model_eval(values, order)
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