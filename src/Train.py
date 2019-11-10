import tensorflow as tf 
import Pipeline

def train(model, input_data, epochs, save_every, batch_size, others):
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
	avg_losses = []
	with tf.Session() as sess:
		# tokenize, vectorize, and batch data
		data_batches, label_batches = [], []
		for i in range(len(input_data) // batch_size):
			end_index = min(len(input_data), (i+1)*batch_size)
			tokenized = [tokenize(headline) for headline in input_data[i*batch_size:end_index]]
			data = [vectorize(words) for words in tokenized]
			labels = [label(words) for words in tokenized]
			data_batches.append(data)
			label_batches.append(labels)
		num_batches = len(data_batches)

		# some weight initialization/other setup required here, I think?

		# train the damn thing
		for epoch in range(epochs):
			for i in range(num_batches):
				epoch_loss += model.train_step(data_batches[i], label_batches[i], sess)
			avg_losses.append(epoch_loss / num_batches) # append losses 
			if (epoch + 1) % save_every == 0:
				# save model, idk how

	return avg_losses