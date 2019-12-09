import tensorflow as tf 
from Pipeline import *
from AggregateModel import AggregateModel
from LSTMLanguageModel import nlpModel
from dataAggregation import getNewsData
from dataAggregation import getStock
from LinearRegressionModel import model_eval_batch, model_eval
import glob
import json
import numpy as np


def train(input_data, validation_data, epochs, save_every, batch_size):
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
	for d in input_data:
		headlines = getNewsData(d)
		nlp_input += [merge_headlines(headlines)]

		# unsure if should be an array here
		tsa_input += [getStock(d, "aa.us")]

	
	avg_losses = []

	# tokenize, vectorize, and batch data
	nlp_data_batches, tsa_data_batches, label_batches = [], [], []


	if (len(nlp_input) != len(tsa_input)):
		print("the inputs should be the same size")


	# length of each headline
	headline_len = 100

	# process the nlp data
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


	print(label_batches[0])
	if (len(nlp_data_batches) != len(tsa_data_batches)):
		print("the data batches should be the same size")


	num_batches = len(nlp_data_batches)

	# Preprocess the validation data as well

	nlp_validation_data = []
	tsa_validation_data = []
	for d in validation_data:
		headlines = getNewsData(d)
		nlp_validation_data += [merge_headlines(headlines)]

		# unsure if should be an array here
		tsa_validation_data += [getStock(d, "aa.us")]

	tokenized = [tokenize(headline) for headline in nlp_validation_data]
	nlp_validation_data = [tokenized_to_numerized(words, token_to_index)[:-1] for words in tokenized]

	tsa_validation_labels = [t[-1] for t in tsa_validation_data]
	print("labels")
	print(tsa_validation_labels)
	tsa_validation_data = [t[:-1] for t in tsa_validation_data]
	#
	# validation_labels, validation_masks = list(zip(*[label(words, index_to_token) for words in tokenized]))
	#



	sess = tf.Session()

    
	# load nlp model
	nlp_model_hyperparameters = {'input_length': 100, 'vocab_size': 16399, 'rnn_size': 256, 'learning_rate': 1e-4, 'embedding_size': 300}
	nlp_model = nlpModel(hyper_parameters=nlp_model_hyperparameters)
	nlpModel.load_model(nlp_model, sess, save_name="language_model")

	# fake tsa model
	tsa_model = lambda x: None
	tsa_callback = model_eval_batch


	model = AggregateModel(nlp_model=nlp_model, tsa_model=tsa_model, tsa_in_tf=False)

	sess.run(tf.global_variables_initializer())

	# train the damn thing
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

	train_dates = ["2015-07-29" for _ in range(32)]
	test_dates = ["2015-07-29" for _ in range(32)]

	epochs = 100
	save_every = 10
	batch_size = 32

	train(train_dates, test_dates, epochs, save_every, batch_size)

if __name__ == '__main__':
	main()
