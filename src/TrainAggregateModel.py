import tensorflow as tf 
from Pipeline import *
from AggregateModel import AggregateModel
from LSTMLanguageModel import nlpModel
import glob
import json
import numpy as np


def train(input_data, validation_data, epochs, save_every, batch_size):
	"""
	Trains the model on the given input data for a number of epochs, with options to control batch sizes and how often to save.

	:param model: The model to train on, using the API in BaseModel
	:param input_data: The input data in the form of news headlines
	:param epochs: The number of epochs to train for
	:param save_every: The frequency (in epochs) with which to save the model by calling model.save_model()
	:param batch_size: The batch size with which to train the model
	:param others: other parameters as needed
	:return: a list of average training loss per epoch
	"""

	# Build a map from token to index
	# with open("vocabulary.txt", 'r') as vocab_file:
	# 	index_to_token = vocab_file.read().split("\n")

	
	arr = getNewsData("2015-07-29")
	merge_headlines(arr)


	token_to_index = {word: index for index, word in enumerate(index_to_token)}

	avg_losses = []

	# tokenize, vectorize, and batch data
	data_batches, label_batches, label_mask_batches = [], [], []

	for i in range(len(input_data) // batch_size):
		end_index = min(len(input_data), (i + 1) * batch_size)
		tokenized = [tokenize(headline) for headline in input_data[i * batch_size:end_index]]
		data = [tokenized_to_numerized(words, token_to_index)[:-1] for words in tokenized]
		labels = [label(words, index_to_token) for words in tokenized]
		data_batches.append(data)
		batch_labels, batch_masks = list(zip(*labels))
		label_batches.append(batch_labels)
		label_mask_batches.append(batch_masks)

	num_batches = len(data_batches)

	# # Preprocess the validation data as well
	# tokenized = [tokenize(headline) for headline in validation_data]
	# validation_data = [tokenized_to_numerized(words, token_to_index)[:-1] for words in tokenized]
	# validation_labels, validation_masks = list(zip(*[label(words, index_to_token) for words in tokenized]))

	sess = tf.Session()

    
	# load nlp model
	nlp_model_hyperparameters = {'input_length': 100, 'vocab_size': 16399, 'rnn_size': 256, 'learning_rate': 1e-4, 'embedding_size': 300}
	nlp_model = nlpModel(hyper_parameters=nlp_model_hyperparameters)
	nlpModel.load_model(nlp_model, sess, save_name="language_model")

	# fake tsa model
	tsa_model = lambda x: None
	tsa_callback = lambda x : [[0]]


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
			loss_value, step = model.train_step(data_batches[i], [[1], [2], [3], [4]], [[2], [3], [4], [5]], sess, tsa_callback)
			epoch_loss += loss_value
			print(f"Loss on training step {step}: {loss_value}")

		print(f"Training Loss: {epoch_loss / num_batches}")
		print(f"Validation Loss: {model.get_loss(validation_data, validation_labels, validation_masks, sess)}")

		# Add the average loss on this epoch to the average losses list
		avg_losses.append(epoch_loss / num_batches)

		# if (epoch + 1) % save_every == 0:
		# 	# Save the model every <save_every> epochs
		# 	model.save_model(sess, "aggregate_model")

		print("===============")

	return avg_losses

def main():
	# Get the data
	train_headlines = ["millennials scare stick", "hyam exdivs lottery waterproof", "exdivs lottery kalvista"]
	test_headlines = ["millennials scare stick", "hyam exdivs lottery waterproof", "exdivs lottery kalvista"]
	time_series_inputs = [[1,2,3,4,5]]
	
	train(train_headlines, test_headlines, 100, 1, 1)

if __name__ == '__main__':
	main()
