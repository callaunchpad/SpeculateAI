import tensorflow as tf 
from Pipeline import *
from AggregateModel import AggregateModel
from LSTMLanguageModel import nlpModel
from dataAggregation import getNewsData3
from dataAggregation import getStock
from LinearRegressionModel import model_eval_batch, model_eval
import glob
import json
import pandas as pd
import numpy as np

def train_validation_split(tsa_input, nlp_inputs, train_ratio=0.8):
	# Assert same length data
	assert len(tsa_input) == len(nlp_inputs), "Passed in data vectors must be same length"

	# Pick random indices in the input data
	num_valid = int(len(tsa_input) * (1 - train_ratio))

	# Select random indices
	validation_indices = set(np.random.choice(list(range(len(tsa_input))), size=num_valid, replace=False))

	# Walk through list and separate out indices
	tsa_train, nlp_train, tsa_test, nlp_test = [], [], [], []

	for i in range(len(tsa_input)):
		if i in validation_indices:
			tsa_test.append(tsa_input[i])
			nlp_test.append(nlp_inputs[i])
		else:
			tsa_train.append(tsa_input[i])
			nlp_train.append(nlp_inputs[i])

	return tsa_train, nlp_train, tsa_test, nlp_test


def train(epochs, save_every, batch_size):
	"""
	Trains the model on the given input data for a number of epochs, with options to control batch sizes and how often to save.

	:param model: The model to train on, using the API in BaseModel
	:param input_data: The input data in the form of dates for each timeseries prediction
	:param epochs: The number of epochs to train for
	:param save_every: The frequency (in epochs) with which to save the model by calling model.save_model()
	:param batch_size: The batch size with which to train the model
	:param others: other parameters as needed
	:return: a list of average training loss per epoch
	"""

	# Build a map from token to index
	with open("vocabulary.txt", 'r') as vocab_file:
		index_to_token = vocab_file.read().split("\n")
	
	token_to_index = {word: index for index, word in enumerate(index_to_token)}

	nlp_input = []
	tsa_input = []

	stocks = glob.glob("../data/Stocks/*.txt")
	min_date, max_date = '2015-12-09', '2017-12-09'
	date_range = pd.date_range(min_date, max_date, freq="D").tolist()

	print(f"Loading data from {len(stocks)} stocks")
	for stock in stocks[:100]:
		for date in date_range:
			headlines = getNewsData3(str(date)[:10], "title", 3)
			# Ensure we have sufficient data here
			if headlines == []:
				continue

			# Get the stock date from the file path
			stock_string = stock.split("/")[-1][:-4]

			stock_data = getStock(date, stock_string)

			if stock_data == []:
				continue

			tsa_input += [stock_data]
			nlp_input += [merge_headlines(headlines)]

	# Split into validation and training
	tsa_input, nlp_input, tsa_validation_data, nlp_validation_data = train_validation_split(tsa_input, nlp_input)

	print(f"TSA Inputs: {len(tsa_input)}")
	exit(0)
	avg_losses = []

	# tokenize, vectorize, and batch data
	nlp_data_batches, tsa_data_batches, label_batches = [], [], []

	# length of each headline
	headline_len = 100

	# this should work
	for i in range(len(nlp_input) // batch_size):
		end_index = min(len(nlp_input), (i + 1) * batch_size)
		tokenized = [tokenize(headline, arr_len=headline_len) for headline in nlp_input[i * batch_size:end_index]]
		data = [tokenized_to_numerized(words, token_to_index)[:-1] for words in tokenized]
		nlp_data_batches.append(data)

		# process the tsa data
		tsa_in = [tsa[:-1] for tsa in tsa_input[i * batch_size:end_index]]
		tsa_data_batches.append(tsa_in)

		batch_labels = [t[-1] for t in tsa_in]
		batch_labels = np.array(batch_labels)
		batch_labels = np.reshape(batch_labels, (-1, 1))
		label_batches.append(batch_labels)

	if (len(nlp_data_batches) != len(tsa_data_batches)):
		print("the data batches should be the same size")

	num_batches = len(nlp_data_batches)


	tokenized = [tokenize(headline) for headline in nlp_validation_data]
	nlp_validation_data = [tokenized_to_numerized(words, token_to_index)[:-1] for words in tokenized]

	tsa_validation_labels = [t[-1] for t in tsa_validation_data]
	tsa_validation_data = [t[:-1] for t in tsa_validation_data]

	# Build the tensorflow session
	sess = tf.Session()

	# load nlp model
	nlp_model_hyperparameters = {'input_length': 100, 'vocab_size': 16399, 'rnn_size': 256, 'learning_rate': 1e-4, 'embedding_size': 300}
	nlp_model = nlpModel(hyper_parameters=nlp_model_hyperparameters)
	nlpModel.load_model(nlp_model, sess, save_name="language_model")

	# fake tsa model
	tsa_callback = model_eval_batch


	# Build the aggregate model
	model = AggregateModel(nlp_model=nlp_model, tsa_model=None, tsa_in_tf=False)

	sess.run(tf.global_variables_initializer())


	print(f"Training model on {num_batches} batches for {epochs} epochs...")
	for epoch in range(epochs):
		print("\n\n===============")
		print(f"Epoch {epoch}")
		print("===============")
		epoch_loss = 0

		# Loop over batches
		for i in range(num_batches):
			loss_value, step = model.train_step(nlp_data_batches[i], tsa_data_batches[i], label_batches[i], sess, tsa_callback=tsa_callback)
			epoch_loss += loss_value
			print(f"Loss on training step {step}: {loss_value}")

		print(f"Training Loss: {epoch_loss / num_batches}")
		print(f"Validation Loss: {model.get_loss(nlp_validation_data, tsa_validation_data, tsa_validation_labels, sess, tsa_callback=tsa_callback)}")

		# Add the average loss on this epoch to the average losses list
		avg_losses.append(epoch_loss / num_batches)

		if (epoch + 1) % save_every == 0:
			# Save the model every <save_every> epochs
			model.save_model(sess, "aggregate_model")

		print("===============")

	return avg_losses

def main():
	# Get the data
	# train_headlines = ["millennials scare stick", "hyam exdivs lottery waterproof", "exdivs lottery kalvista"]
	# test_headlines = ["millennials scare stick", "hyam exdivs lottery waterproof", "exdivs lottery kalvista"]
	# time_series_inputs = [[1,2,3,4,5]]

	epochs = 100
	save_every = 10
	batch_size = 32

	train(epochs, save_every, batch_size)

if __name__ == '__main__':
	main()
