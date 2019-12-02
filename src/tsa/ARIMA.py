import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot as plt
from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error
from features import *
import pickle
from pyramid.arima import auto_arima
import glob

def main():
    data = pd.read_csv("../../data/DJIA_table.csv", header=0, index_col=0)
    data = data.reindex(index=data.index[::-1])
    series = data["Close"]
    # model_eval(series)
    stock_dict = read_entire_market()
    save_all_models(stock_dict)
    # models, percentage, error, percentage_up, error_up, percentage_down, error_down = train_on_market(stock_dict)
    # print("Percentage correctly labeled: ", percentage, "Total error: ", error)
    # print("Percentage when market went up: ", percentage_up, "Total error: ", error_up)
    # print("Percentage when market went down: ", percentage_down, "Total error: ", error_down)



#Reads the entire stocks dataset into a dictionary with key: stock name, value: values array
def read_entire_market():
    stock_path = "../../data/Stocks/"
    all_stocks = glob.glob(stock_path + "/*.txt")
    stock_dict = {}
    count = 0
    for filename in all_stocks:
        count += 1
        try:
            stock = pd.read_table(filename.replace(".us.txt", ""), delimiter=',', header=0, index_col=0)
            data = list(reversed(stock["Close"].values))
            stock_dict[filename] = data
        except:
            continue
    print("count", count)
    return stock_dict
#Loop to save all the arima models for each stock in the Stocks dataset
def save_all_models(stock_dict):
    for stock in stock_dict:
        try:
            series = stock_dict[stock]
            model = auto_arima(series[:len(series -1)], disp=-1, suppress_warnings=True)
            save_model(model, stock)
        except:
            print("Error in saving model for stock: ", stock)
            continue

#Loop to load all saved models
def load_all_models():
    model_path = "arima_models/"
    all_models = glob.glob(model_path + "/*.sav")
    model_dict = {}
    for filename in all_models:
        model = pickle.load(open(filename, "rb"))
        stock_name = filename.replace(".sav", "")
        model_dict[stock_name] = model
    return model_dict

#Training loop to determine classification accuracy on entire market
def train_on_market(stock_dict):
    models = {}
    iterations = 0
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
        print("Iteration: ", iterations)
        iterations += 1
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

#Evaluates the classification accuracy of a model with one forecast
def model_eval(values):
    train, test = values[0:len(values)-1], values[len(values)-1]
    predictions = list()
    #labels 0 is go down, 1 is go up
    previous = train[len(train)-1]
    model = auto_arima(train, disp=-1, suppress_warnings=True)
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
    error = (abs(test - prediction)/test) * 100
    print("Predicted: ", prediction, "Actual: ", test, "Previous: ", previous, "Percent error: ", error)
    print("Pred label: ", classification, "True label: ", true, "Correct? ", yes)
    return model, correct, error, true

#Given an already fitted model, make a forecast of the value of the stock at the next day
def forecast(model):
    return model.predict(1)[0]


#Test for projecting multiple time steps with arima model
def projection_test(values):
    split = int(0.75 * len(values))
    train, test = values[0:split], values[split:]
    model = auto_arima(train, disp=-1, suppress_warnings=True)
    prediction = model.predict(len(test))
    plt.plot(test)
    plt.plot(prediction)
    plt.show()

#Test for saving a single model
def save_model(model, stock_name):
    filename = "arima_models/" + stock_name + ".sav"
    pickle.dump(model, open(filename, 'wb'))

#Test for loading a single model
def load_model(stock_name):
    filename = "arima_models/" + stock_name + ".sav"
    model = pickle.load(open(filename, 'rb'))
    return model


#Computes first order differencing of a series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return pd.Series(diff)

#Test for saving and loading a single model
def test_save_load(series):
    model = auto_arima(series[:len(series) - 1], disp=-1, suppress_warnings=True)
    save_model(model, "DJIA")
    load_model("DJIA")
    print(model.predict(1)[0])

if __name__== "__main__":
    main()