from models import *
from numpy import array
from lstm_regress import lstmTSAModel
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import os

n_steps = 20

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

		# for regression:
		seq_y = [seq_x[end_ix-1]]

		# for classification:
		# seq_y = [1, 0] # fall
		# if seq_x[end_ix-1] > seq_x[end_ix-2]:
		# 	seq_y = [0, 1] # rise

		X.append(seq_x) 
		y.append(seq_y) # if classification: want this to be 1 or 0, might need to one-hot encode your label [fall, rise]
	return array(X), array(y)


def train(model, input_data, validation_data, epochs, save_every, batch_size):
	"""
	Trains the model on the given input data for a number of epochs, with options to control batch sizes and how often to save.
	:param model: The model to train on, using the API in BaseModel
	:param input_data: The input data in the form of news close
	:param epochs: The number of epochs to train for
	:param save_every: The frequency (in epochs) with which to save the model by calling model.save_model()
	:param batch_size: The batch size with which to train the model
	:param others: other parameters as needed
	:return: a list of average training loss per epoch
	"""

	avg_losses = []

	# tokenize, vectorize, and batch data
	data_batches, label_batches = [], []

	# split sequence run here
	# start splicing
	# for loop is indexing into big X return of split_sequence
	X, y = split_sequence(input_data, n_steps)

	for i in range(len(X) // batch_size):
		end_index = min(len(X), (i + 1) * batch_size)
		#tokenized = [split_sequence(input_data)[0] for headline in input_data[i * batch_size:end_index]] ### main line for splitting
		#data = [tokenized_to_numerized(words, token_to_index)[:-1] for words in tokenized]

		#incorporate shuffling, and batch the indices to access the data batches and label batches
		data = [d for d in  X[i * batch_size:end_index]]
		label = [l for l in y[i * batch_size:end_index]]
		#labels = [label(words, index_to_token) for words in tokenized]
		data_batches.append(data)
		label_batches.append(label)
		#batch_labels, _ = list(zip(*labels))
		#label_batches.append(batch_labels)
		#label_mask_batches.append(batch_masks)

	num_batches = len(data_batches)
	#print(np.array(data_batches[0]).shape)
	#print(num_batches)

	# Preprocess the validation data as well
	validation_data, validation_labels = split_sequence(validation_data, n_steps)
	# tokenized = [tokenize(headline) for headline in validation_data]
	# validation_data = [tokenized_to_numerized(words, token_to_index)[:-1] for words in tokenized]
	# validation_labels, validation_masks = list(zip(*[label(words, index_to_token) for words in tokenized]))

	sess = tf.Session()

	# some weight initialization/other setup required here, I think?
	sess.run(tf.global_variables_initializer())

	# train the damn thing
	print(f"Training model on {num_batches} batches for {epochs} epochs...")
	for epoch in range(epochs):
		print("\n\n===============")
		print(f"Epoch {epoch}")
		print("===============")
		epoch_loss = 0

		# Loop over batches
		# print("before train_step!!!")
		for i in range(num_batches):
			#print(data_batches[i])
			#print(np.array(data_batches[i]).shape)
			loss_value, step = model.train_step(np.array(data_batches[i]), np.array(label_batches[i]), sess)
			epoch_loss += loss_value
			print(f"Loss on training step {step}: {loss_value}")

		print(f"Training Loss: {epoch_loss / num_batches}")
		print(f"Validation Loss: {model.get_loss(validation_data, validation_labels, sess)}")

		# Add the average loss on this epoch to the average losses list
		avg_losses.append(epoch_loss / num_batches)

		if (epoch + 1) % save_every == 0:
			# Save the model every <save_every> epochs
			model.save_model(sess, "lstm_tsa_model")

		print("===============")

	# model.load_model(sess, "lstm_tsa_model")

	# calculate model accuracy
	correct = 0
	outputs_list = []
	labels_list = []
	for i in range(num_batches):
		outputs = model.predict(data_batches[i], sess)
		for j in range(len(outputs)):
			print("=========================================================")
			output_value = outputs[j][0][0]
			label_value = label_batches[i][j][0]
			print(output_value)
			print(label_value)
			outputs_list.append(output_value)
			labels_list.append(label_value)
			if output_value == label_value:
				correct += 1
	accuracy = (1.0 * correct)/ (1.0 * num_batches * len(data_batches[i]))
	print(f"n_steps: {n_steps}")
	print(f"Model accuracy: " + str(accuracy) + "%")

	# plot data
	r = list(range(num_batches * len(data_batches[i])))
	predicted_price, = plt.plot(r, outputs_list, color='green', label="predicted price")
	actual_price, = plt.plot(r, labels_list, color='red', label="actual price")
	plt.legend([predicted_price, actual_price], ["predicted price", "actual price"])
	plt.xlabel("Training Time Steps")
	plt.ylabel("Closing Prices")
	plt.show()

	return avg_losses

def main():
	# Get the data
	x = pd.read_csv("../../data/test/Data/Stocks/a.us.txt")
	split = int(len(x["Close"]) * (1 - 20/100))
	close = x["Close"]
	close_train = x["Close"][:split].reset_index(drop=True)
	close_test = x["Close"][split:].reset_index(drop=True)

	print(f"Loaded {len(close)} close...")
	# Split the data into train and validation
	train_percentage = 0.8

	train_close = close[:int(train_percentage * len(close))].reset_index(drop=True)
	test_close = close[int(train_percentage * len(close)):].reset_index(drop=True)

	# Build the model
	model_hyperparameters = {
		'input_length': n_steps,
		'rnn_size': 256,
		'learning_rate': 1e-4,
		'forecast_horizon': 1
	}

	model = lstmTSAModel(hyper_parameters=model_hyperparameters)

	avg_losses = train(model, train_close, test_close, 100, 1, 32)
	print(avg_losses)

if __name__ == '__main__':
	main()