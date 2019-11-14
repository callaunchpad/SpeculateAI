from models import *
from numpy import array
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

test_percent = 20 # Percent of data used for verification

df_DJIA = pd.read_csv("../../data/DJIA_table.csv")
split = int(len(df_DJIA["Close"]) * (1 - test_percent/100))

close = df_DJIA["Close"]
close_train = df_DJIA["Close"][:split]
close_test = df_DJIA["Close"][split:]

# x = pd.read_csv("../../data/test/Data/Stocks/a-t/a.us.txt")
# split = int(len(x["Close"]) * (1 - 20/100))
# close_train = x["Close"][:split].reset_index(drop=True)
# close_test = x["Close"][split:].reset_index(drop=True)

# print(close_train)
# print(close_test)

# split a univariate sequence into samples
# n_steps is a hyperparameter, tells you how far back in time to look for prediction
def split_sequence(sequence, n_steps): 
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x = sequence[i:end_ix]
		seq_y = [1, 0] # fall
		if seq_x[end_ix-1] > seq_x[end_ix-2]:
			seq_y = [0, 1] # rise
		X.append(seq_x) 
		y.append(seq_y) # want this to be 1 or 0, might need to one-hot encode your label [fall, rise]
	return array(X), array(y)

def model_eval(train_data, test_data, p):

	n_steps = p[0] # n_steps = how far back in time to look for prediction

    # def feature_list(data):

    #     features = []

    #     bb1, bb2 = BB(data, window)
    #     bb1, bb2 = bb1, bb2

    #     features.extend(get_lags(data, lag_window))
    #     features.append(MA(data, window)[lag_window:])
    #     features.append(EMA(data, window).tolist()[lag_window:])
    #     features.append(MACD(data).tolist()[lag_window:])
    #     features.append(bb1[lag_window:])
    #     features.append(bb2[lag_window:])
    #     features.append([bb1[i] - bb2[i] for i in range(lag_window, len(bb1))])
    #     features.append(RSI(data, window).tolist()[lag_window:])
    #     features.append(VMA(data, window).tolist()[lag_window:])

    #     return features

    # define input sequence
	raw_seq = train_data
	# split into samples
	X, y = split_sequence(raw_seq, n_steps)
	# reshape from [samples, timesteps] into [samples, timesteps, features]
	n_features = 1
	X = X.reshape((X.shape[0], X.shape[1], n_features))

	model = Sequential()
	model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
	model.add(Dense(2, activation='softmax'))
	model.compile(optimizer='adam', loss='categorical_crossentropy')
	model.fit(X, y, epochs=200, verbose=0)

	labels = get_labels(test_data)
	predictions = []
	for i in range(len(test_data-1)):
		x_input = close[split+i-n_steps+1:split+i+1]
		# print("iteration: " + str(i))
		x_input = x_input.reshape(1, n_steps, n_features)
		predictions.append(np.argmax(model.predict(x_input, verbose=0)))

	# print(predictions)
	# print(labels)
	# print(len(predictions))
	# print(len(labels))

	correct = 0
	for i in range(len(labels)):
		if predictions[i] == labels[i]:
			correct += 1
	accuracy = (1.0 * correct)/ (1.0 * len(predictions))
	print(f"n_steps: {n_steps}")
	print(f"Model accuracy: " + str(accuracy) + "%")
	return accuracy


x = list(range(60, 100, 10))
plt.plot(x, [model_eval(close_train, close_test, [param]) for param in x], color='green')
plt.xlabel("n_steps")
plt.ylabel("Accuracy (%)")
plt.show()

# # demonstrate prediction
# x_input = close_test[:n_steps] # x_input should be the size of a sample
# print(x_input)
# x_input = x_input.reshape((1, n_steps, n_features))
# yhat = model.predict(x_input, verbose=0)
# print(yhat)