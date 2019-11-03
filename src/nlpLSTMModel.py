import tensorflow as tf

"""
This is the basic structure of the models in this project.
In this context, model is a substructure of the overall aggregated graph
"""

class nlpModel:
    def __init__(self, hyper_parameters={}):
        """
        Sets up any hyper-parameters that we may need for the model and builds the graph
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
            self.input_length = hyper_parameters['input_length']
            self.vocab_size = hyper_parameters['vocab_size']
            self.rnn_size = hyper_parameters['rnn_size']
            self.learning_rate = hyper_parameters['learning_rate']
            
            self.inputs = tf.placeholder(tf.int32, shape=[None, input_length])
            self.labels = tf.placeholder(tf.float32, shape=[None, vocab_size])
            
            self.labels_mask = tf.placeholder(tf.int32, shape=[None, vocab_size])

            self.embedding = tf.get_variable("Embedding", shape=[vocab_size, rnn_size])
            input_emb = tf.nn.embedding_lookup(embedding, self.input_num)

            num_cells = 1        
            
            cells = [tf.nn.rnn_cell.LSTMCell(rnn_size) for i in range(num_cells)]
            multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
            
            self.output, states = tf.nn.dynamic_rnn(multi_cell, inputs=input_emb, dtype=tf.float32)
            
            output_logits = tf.layers.dense(self.output, vocab_size, name="Output_Projection")
            self.loss = tf.losses.sparse_softmax_cross_entropy(tf.cast(self.labels, tf.int32), output_logits)
            
            optimizer = tf.train.AdamOptimizer(self.learning_rate)        
            self.train_op = optimizer.minimize(self.loss)

        return

    def train_step(self, inputs, labels, sess):
        """
        Performs a training step using self.train_op in the passed session.

        :param inputs: The inputs to forward propagate
        :param labels: The outputs used in computing the loss function
        :param sess: The tensorflow session used for evaluating the training operation
        :return: The loss on <inputs> and <outputs>
        """

        feed_dict = {
            self.inputs: inputs,
            self.labels: labels
        }

        loss_value, _ = sess.run([self.loss, self.train_op], feed_dict=feed_dict)

        return loss_value

    def get_loss(self, inputs, labels, sess):
        """
        Evaluates the loss on the given inputs and outputs in the given TF session

        :param inputs: The inputs to forward propagate
        :param labels: The outputs used in computing the loss function
        :param sess: The tensorflow session used for evaluating the loss operation
        :return: The loss on <inputs> and <outputs>
        """

        feed_dict = {
            self.inputs: inputs,
            self.output: labels
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
