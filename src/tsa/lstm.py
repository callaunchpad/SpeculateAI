from models import *
from numpy import array
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

test_percent = 20 # Percent of data used for verification

df_DJIA = pd.read_csv("../../data/DJIA_table.csv")
split = int(len(df_DJIA["Close"]) * (1 - test_percent/100))

close = df_DJIA["Close"]
close_train = df_DJIA["Close"][:split]
close_test = df_DJIA["Close"][split:]

print(close_train)
print(close_test)

# split a univariate sequence into samples
def split_sequence(sequence, n_steps): 
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		# seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		seq_x = sequence[i:end_ix]
		seq_y = [1, 0] # fall
		if seq_x[end_ix-1] > seq_x[end_ix-2]:
			seq_y = [0, 1] # rise
		X.append(seq_x) 
		y.append(seq_y) # want this to be 1 or 0, might need to one-hot encode your label [fall, rise]
	return array(X), array(y)
 
# define input sequence
raw_seq = close_train
# choose a number of time steps
n_steps = 20
# n_steps = len(close_test)
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy')
# fit model
model.fit(X, y, epochs=200, verbose=0)


labels = get_labels(close_test)
predictions = []
for i in range(len(close_test)):
	x_input = close[split+i-n_steps+1:split+i+1]
	print("iteration: " + str(i))
	x_input = x_input.reshape(1, n_steps, n_features)
	predictions.append(np.argmax(model.predict(x_input, verbose=0)))
print(predictions)
print(labels)

print(len(predictions))
print(len(labels))

correct = 0
for i in range(len(predictions)):
	if predictions[i] == labels[i]:
		correct += 1
print(correct/len(predictions))

# # demonstrate prediction
# x_input = close_test[:n_steps] # x_input should be the size of a sample
# print(x_input)
# x_input = x_input.reshape((1, n_steps, n_features))
# yhat = model.predict(x_input, verbose=0)
# print(yhat)