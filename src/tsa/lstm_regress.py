import tensorflow as tf
import os

"""
This is the basic structure of the models in this project.
In this context, model is a substructure of the overall aggregated graph
"""

class lstmTSAModel:
    def __init__(self, hyper_parameters={}):
        """
        Sets up any hyper-parameters that we may need for the model and builds the graph
        Hyper Parameters:
            input_length (n_steps)
            rnn_size
            learning_rate
            forecast_horizon
        """
        # The following sets the tensorflow scope for this particular model
        # It is important that the model have one of a few scopes:
        #       - NLP: for models evaluating the NLP branch
        #       - TSA: for models evaluating the TSA branch
        #       - DOWN: for models evaluating the combined time series and nlp outputs
        # These scopes are used in aggregation to allow for end-to-end training or downstream fine tuning
        scope = "TSA"
        self.build_graph(scope, hyper_parameters)

    def build_graph(self, scope="TSA", hyper_parameters={}):
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
            #self.num_hidden_layers = hyper_parameters['num_hidden_layers']
            self.input_length = hyper_parameters['input_length']
            self.rnn_size = hyper_parameters['rnn_size']
            self.learning_rate = hyper_parameters['learning_rate']
            self.forecast_horizon = hyper_parameters['forecast_horizon'] # set this to 1
            
            self.inputs = tf.placeholder(tf.int32, shape=[None, self.input_length])
            self.labels = tf.placeholder(tf.float32, shape=[None, self.forecast_horizon]) # 2 bc for classification, 1 bc for regression

            num_cells = 1 # potential hyperparameter
            
            cells = [tf.nn.rnn_cell.LSTMCell(hyper_parameters['rnn_size']) for _ in range(num_cells)]
            multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

            lm_cell = tf.nn.rnn_cell.DropoutWrapper(multi_cell, output_keep_prob=0.5)

            # batch of time series, and each times serives are a vector, pad it with an extra dim, tf.expand_dim (append to the end)
            input_extra_dim = tf.cast(tf.expand_dims(self.inputs, 1), tf.float32)

            rnn_out, self.out_states = tf.nn.dynamic_rnn(lm_cell, inputs=input_extra_dim, dtype=tf.float32) #inputs needs to be 3D???
            
            # output is how many values you're regressing on (it's on the closing price) 
            self.output = tf.layers.dense(rnn_out, self.forecast_horizon, name="Output_Projection")

            self.loss = tf.keras.losses.MSE(self.labels, self.output) # loss for regression
            #self.loss = tf.keras.losses.categorical_crossentropy(self.labels, self.output) # loss for classification

            self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, 500, 0.983)

            optimizer = tf.train.AdamOptimizer(self.learning_rate)        
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

            vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self.saver = tf.train.Saver(var_list=vars)

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
            self.labels: labels,
        }

        loss_value, global_step, _ = sess.run([self.loss, self.global_step, self.train_op], feed_dict=feed_dict)

        return loss_value, global_step

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
            self.labels: labels,
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