import tensorflow as tf 
#from Pipeline import *
from lstm_regress import lstmTSAModel
from models import *
import glob
import json
import numpy as np
import os

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

	# # Build a map from token to index
	# with open("./vocabulary.txt", 'r') as vocab_file:
	# 	index_to_token = vocab_file.read().split("\n")

	# token_to_index = {word: index for index, word in enumerate(index_to_token)}

	avg_losses = []

	# tokenize, vectorize, and batch data
	data_batches, label_batches = [], []

	#split sequence run here
	#start splicing
	#for loop is indexing into big X return of split_sequence
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

	# Preprocess the validation data as well
	validation_data, validation_labels = split_sequence(validation_data, n_steps)
	# tokenized = [tokenize(headline) for headline in validation_data]
	# validation_data = [tokenized_to_numerized(words, token_to_index)[:-1] for words in tokenized]
	# validation_labels, validation_masks = list(zip(*[label(words, index_to_token) for words in tokenized]))

	sess = tf.Session()

	# some weight initialization/other setup required here, I think?
	sess.run(tf.global_variables_initializer())

	#model.load_model(sess, "language_model")
	#start = input("Sentence to be completed: ")
	#print(f"Sample generated sentence: {generate_headline(model, token_to_index, index_to_token, sess, starter=start)}")


	# train the damn thing
	print(f"Training model on {num_batches} batches for {epochs} epochs...")
	for epoch in range(epochs):
		print("\n\n===============")
		print(f"Epoch {epoch}")
		print("===============")
		epoch_loss = 0

		# Loop over batches
		for i in range(num_batches):
			loss_value, step = model.train_step(data_batches[i], label_batches[i], sess)
			epoch_loss += loss_value
			print(f"Loss on training step {step}: {loss_value}")

		print(f"Training Loss: {epoch_loss / num_batches}")
		print(f"Validation Loss: {model.get_loss(validation_data, validation_labels, sess)}")
		#print(f"Sample generated sentence: {generate_headline(model, token_to_index, index_to_token, sess)}")

		# Add the average loss on this epoch to the average losses list
		avg_losses.append(epoch_loss / num_batches)

		if (epoch + 1) % save_every == 0:
			# Save the model every <save_every> epochs
			model.save_model(sess, "language_model")

		print("===============")

	return avg_losses

# def generate_headline(model, token_to_num, num_to_token, session, length=20, lstm_length=100, starter=""):
# 	"""
# 	Generates a headline to test the model using the starter sentence
# 	:param model: The model to use for language modeling
# 	:param token_to_num: A mapping from tokens to numeric values
# 	:param num_to_token: A mapping from numerized values to tokens
# 	:param starter: The start of the sentence we would like to complete
# 	:return: A generated sentence
# 	"""

# 	# Tokenize the starter input
# 	tokenized = starter.split(" ") if starter != "" else []

# 	# Numerize this starter
# 	current_sentence = [token_to_num["START"]] + tokenized_to_numerized(tokenized, token_to_num)

# 	while len(current_sentence) < length:
# 		# Pad the input for the LSTM
# 		padded_input = current_sentence[:length] + [token_to_num["PAD"]] * (lstm_length - len(current_sentence) - 2)
# 		padded_input = np.reshape(np.array(padded_input), (1, -1))

# 		# Run the model on this padded input to predict the next value
# 		model_logits = model.predict(padded_input, session)
# 		model_logits = np.squeeze(model_logits) # Remove extraneous dimensions

# 		# Find the logits for the appropriate input
# 		next_token_logits = model_logits[len(current_sentence) - 1, :]
# 		next_token_logits[current_sentence[-1]] = float('-inf')

# 		# Append the argmax token to the current sentence
# 		current_sentence.append(np.argmax(next_token_logits))


# 	# Put the sentence together
# 	return " ".join([num_to_token[numeric] for numeric in current_sentence])

def main():
	# Get the data
	# test_percent = 20 # Percent of data used for verification

	x = pd.read_csv("../../data/test/Data/Stocks/a.us.txt")
	split = int(len(x["Close"]) * (1 - 20/100))
	close = x["Close"]
	close_train = x["Close"][:split].reset_index(drop=True)
	close_test = x["Close"][split:].reset_index(drop=True)

	# close = []

	# json_files = glob.glob("../data/small_financial_news/*.json")

	# for json_file in json_files:
	# 	# Open the json file and pull the headline
	# 	with open(json_file, encoding='utf-8') as file:
	# 		# Load the title string into headline
	# 		headline = json.loads(file.read())['title']
	# 		# Clean the headline
	# 		headline = clean_headline(headline)
	# 		# Add it to the list of close
	# 		close.append(headline)

	# 		if len(close) > 5000:
	# 			break


	print(f"Loaded {len(close)} close...")
	# Split the data into train and validation
	train_percentage = 0.8

	train_close = close[:int(train_percentage * len(close))]
	test_close = close[int(train_percentage * len(close)):]

	# Build the model
	model_hyperparameters = {
		'n_steps': 5,
		'input_length': 100,
		'rnn_size': 256,
		'learning_rate': 1e-4,
		'forecast_horizon': 1
	}

	model = lstmTSAModel(hyper_parameters=model_hyperparameters)

	train(model, train_close, test_close, 100, 1, 32)

if __name__ == '__main__':
	main()