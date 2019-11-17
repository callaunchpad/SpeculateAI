import tensorflow as tf
import os

"""
This is the basic structure of the models in this project.
In this context, model is a substructure of the overall aggregated graph
"""

class nlpModel:
    def __init__(self, hyper_parameters={}):
        """
        Sets up any hyper-parameters that we may need for the model and builds the graph

        Hyper Parameters:
            input-length
            vocab_size
            rnn_size
            learning_rate
        """
        # The following sets the tensorflow scope for this particular model
        # It is important that the model have one of a few scopes:
        #       - NLP: for models evaluating the NLP branch
        #       - TSA: for models evaluating the TSA branch
        #       - DOWN: for models evaluating the combined time series and nlp outputs
        # These scopes are used in aggregation to allow for end-to-end training or downstream fine tuning
        scope = "NLP"
        self.build_graph(scope, hyper_parameters)

    def build_graph(self, scope="NLP", hyper_parameters={}):
        """
        Builds the computation graph this model will be working with.

        Will build the graph with specific fields:
            - self.inputs: an input tf placeholder that will be called
            - self.output: a tensorflow operation to be evaluated for network output
            - self.labels: a tensorflow placeholder for holding labels during training
            - self.loss: a tensorflow operation that evaluates to the loss given inputs and outputs
            - self.train_op: a training operation we will call during training

        :return: None
        """

        with tf.variable_scope(scope):
            self.input_length = hyper_parameters['input_length'] - 2
            self.vocab_size = hyper_parameters['vocab_size']
            self.rnn_size = hyper_parameters['rnn_size']
            self.learning_rate = hyper_parameters['learning_rate']
            
            self.inputs = tf.placeholder(tf.int32, shape=[None, self.input_length])
            self.labels = tf.placeholder(tf.float32, shape=[None, self.input_length])
            self.labels_mask = tf.placeholder(tf.int32, shape=[None, self.input_length])

            self.embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.rnn_size])
            input_embedding = tf.nn.embedding_lookup(self.embedding, self.inputs)

            num_cells = 2
            
            cells = [tf.nn.rnn_cell.LSTMCell(hyper_parameters['rnn_size']) for _ in range(num_cells)]
            multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

            lm_cell = tf.nn.rnn_cell.DropoutWrapper(multi_cell, output_keep_prob=0.5)
            
            rnn_out, self.out_states = tf.nn.dynamic_rnn(lm_cell, inputs=input_embedding, dtype=tf.float32)
            
            self.output = tf.layers.dense(rnn_out, self.vocab_size, name="Output_Projection")

            self.labels_mask = tf.cast(self.labels_mask, tf.float32)
            self.loss = tf.losses.sparse_softmax_cross_entropy(tf.cast(self.labels, tf.int32), self.output, weights=self.labels_mask)

            self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, 500, 0.983)

            optimizer = tf.train.AdamOptimizer(self.learning_rate)        
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

            vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self.saver = tf.train.Saver(var_list=vars)

        return

    def train_step(self, inputs, labels, labels_mask, sess):
        """
        Performs a training step using self.train_op in the passed session.

        :param inputs: The inputs to forward propagate
        :param labels: The outputs used in computing the loss function
        :param sess: The tensorflow session used for evaluating the training operation
        :return: The loss on <inputs> and <outputs>
        """

        feed_dict = {
            self.inputs: inputs,
            self.labels: labels,
            self.labels_mask: labels_mask
        }

        loss_value, global_step, _ = sess.run([self.loss, self.global_step, self.train_op], feed_dict=feed_dict)

        return loss_value, global_step

    def get_loss(self, inputs, labels, mask, sess):
        """
        Evaluates the loss on the given inputs and outputs in the given TF session

        :param inputs: The inputs to forward propagate
        :param labels: The outputs used in computing the loss function
        :param sess: The tensorflow session used for evaluating the loss operation
        :return: The loss on <inputs> and <outputs>
        """

        feed_dict = {
            self.inputs: inputs,
            self.labels: labels,
            self.labels_mask: mask
        }

        loss_value = sess.run(self.loss, feed_dict=feed_dict)

        return loss_value

    def predict(self, inputs, sess):
        """
        Runs inference on the given inputs in the given tensorflow session

        :param inputs: The inputs to forward propagate through the model's graph
        :param sess: The tensorflow session to run the computation in
        :return: The predicted tensor resulting from forward propagation
        """

        feed_dict = {
            self.inputs: inputs
        }

        outputs = sess.run(self.output, feed_dict=feed_dict)

        return outputs

    def save_model(self, sess, save_name=None):
        """
        Saves the computation graph into a tf save format

        :param sess: The tf session to save a graph from
        :param save_name: The output file name
        :return: None
        """
        print("Saving model...")

        if not os.path.exists("./checkpoints"):
            os.makedirs("./checkpoints")

        if save_name is not None:
            self.saver.save(sess, "./checkpoints/" + save_name)
            return

        self.saver.save(sess, "./checkpoints/saved_model")

    def load_model(self, sess, save_name=None):
        """
        Loads a saved model from tf save format into computational graph

        :param sess: The computation session in which to load the graph
        :param save_name: The name of the checkpoints to load
        :return: None
        """
        print("Loading model...")

        if save_name is not None:
            self.saver.restore(sess, "./checkpoints/" + save_name)
        else:
            self.saver.restore(sess, "./checkpoints/saved_model")
