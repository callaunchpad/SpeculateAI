import tensorflow as tf

"""
This code defines how we aggregate natural language, time series, and output graphs
into a single, cohesive computation graph.

The code below expects objects of the same structure as defined in BaseModel.py
"""

class AggregateModel():
    def __init__(self, nlp_model, tsa_model, downstream_model, combination=tf.concat,
                 combination_args=[1], label_shape=[1], is_classifier=True, tsa_in_tf=True):
        """
        Initializes the full computation graph such that we have a complete aggregated model

        :param nlp_model: The nlp model to use in graph combination
        :param tsa_model: The time series model to use in graph combination
        :param downstream_model: The model that will take in combined outputs of nlp and tsa models
        :param combination: The method by which we combine the nlp and tsa outputs
        :param combination_args: The extra arguments needed for the combination method
        :param label_shape: The shape our labels will take on without batch size
        :param is_classifier: Whether or not the output should be interpreted as classification or regression
        :param tsa_in_tf: Whether or not the time series model is a tensorflow graph
        """

        self.is_classifier = is_classifier
        self.combine_models(nlp_model, tsa_model, downstream_model, combination, combination_args, label_shape, tsa_in_tf)

    def combine_models(self, nlp_model, tsa_model, downstream_model, combination, combination_args, label_shape, tsa_in_tf):
        """
        Builds the computation graph by connecting the component graphs in the appropriate manner

        :param nlp_model: The natural language model used in aggregated graph
        :param tsa_model: The time series model used in aggregated graph
        :param downstream_model: The part of the aggregated graph that receives combined nlp and tsa outputs
        :param combination: The tensorflow operation used to combine the nlp and tsa outputs
        :param combination_args: The extra arguments needed for the combination method
        :param label_shape: The shape our labels will take on without batch size
        :return: None
        """

        # The inputs and output from the nlp model
        self.nlp_inputs = nlp_model.inputs
        self.nlp_outputs = nlp_model.out_states[1][1]

        # The inputs and outputs from the time series model
        if tsa_in_tf:
            self.tsa_inputs = tsa_model.inputs
            self.tsa_outputs = tsa_model.output
        else:
            # If our tsa model is a not a TF graph (e.g. logistic regression)
            # In this case we should just treat its outputs as something we cannot backprop into 
            self.tsa_outputs = tf.placeholder(dtype=tf.float32, shape=[1 , None], name="tsa_outputs")

        # The combination of the nlp and time series outputs
        downstream_input = combination([self.nlp_outputs, self.tsa_outputs], *combination_args)

        # The input the downstream model
        downstream_model.inputs = downstream_input

        # The output of the downstream model
        self.downstream_output = downstream_model.output

        # The labels fed in for training the aggregated model
        self.model_labels = tf.placeholder(shape=[None] + label_shape, dtype=tf.float32, name="model_labels")

        # The loss for the aggregated model
        if self.is_classifier:
            self.loss = tf.losses.softmax_cross_entropy(self.model_labels, self.downstream_output)
        else:
            self.loss = tf.losses.mean_squared_error(self.model_labels, self.downstream_output)

 
        # Setting the trainable variables for the optimizer, originally just fine tuning
        downstream_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="DOWN")

        # The training operation called to perform a gradient step
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss, var_list=downstream_vars)


    def train_step(self, nlp_inputs, tsa_inputs, labels, sess, tsa_callback=None):
        """
        Performs a training step of the trainable variables

        :param nlp_inputs: The inputs to the NLP branch of the model
        :param tsa_inputs: The inputs to the time series branch of the model
        :param labels: The labels on which to evaluate loss and perform gradient calculations
        :param sess: The tensorflow session in which to run the evaluation and backpropagation
        :param tsa_callback: The callback of the TSA model if we are not using a tensorflow implemented
        time series model, this allows us to pass in a generic function to evaluate, for example, an
        ARIMA model
        :return: The loss on this given training step
        """

        if tsa_callback:
            # Get the tsa outputs
            tsa_outputs = tsa_callback(tsa_inputs)

            feed_dict = {
                self.nlp_inputs: nlp_inputs,
                self.tsa_outputs: tsa_outputs,
                self.model_labels: labels
            }

            loss_value, _ = sess.run([self.loss, self.train_op], feed_dict=feed_dict)

        else:
            feed_dict = {
                self.nlp_inputs: nlp_inputs,
                self.tsa_inputs: tsa_inputs,
                self.model_labels: labels
            }

            loss_value, _ = sess.run([self.loss, self.train_op], feed_dict=feed_dict)

        return loss_value

    def get_loss(self, nlp_inputs, tsa_inputs, labels, sess, tsa_callback=None):
        """
        Computes the loss on the given set of inputs and labels

        :param nlp_inputs: The inputs to the NLP branch of the model
        :param tsa_inputs: The inputs to the TSA branch of the model
        :param labels: The labels used in computing the loss
        :param sess: The tensorflow session in which to evaluate the graph
        :param tsa_callback: The callback of the TSA model (see train_step docstring for more info)
        :return: The loss on this given (inputs, labels) set
        """

        if tsa_callback:
            # Get the tsa outputs
            tsa_outputs = tsa_callback(tsa_inputs)

            feed_dict = {
                self.nlp_inputs: nlp_inputs,
                self.tsa_outputs: tsa_outputs,
                self.model_labels: labels
            }

            loss_value = sess.run(self.loss, feed_dict=feed_dict)

        else:
            feed_dict = {
                self.nlp_inputs: nlp_inputs,
                self.tsa_inputs: tsa_inputs,
                self.model_labels: labels
            }

            loss_value = sess.run(self.loss, feed_dict=feed_dict)

        return loss_value


    def predict(self, nlp_inputs, tsa_inputs, sess, tsa_callback=None):
        """
        Runs inference on the aggregated model for given inputs

        :param nlp_inputs: The inputs to the NLP branch of the model
        :param tsa_inputs: The inputs to the TSA branch of the model
        :param sess: The tensorflow session in which to evaluate the graph
        :param tsa_callback: See train_step docstring
        :return: The predicted values after inference
        """

        if tsa_callback:
            # Get the tsa outputs
            tsa_outputs = tsa_callback(tsa_inputs)

            feed_dict = {
                self.nlp_inputs: nlp_inputs,
                self.tsa_outputs: tsa_outputs
            }

            predictions = sess.run(self.model_output, feed_dict=feed_dict)

        else:
            feed_dict = {
                self.nlp_inputs: nlp_inputs,
                self.tsa_inputs: tsa_inputs
            }

            predictions = sess.run(self.model_output, feed_dict=feed_dict)

        return predictions

    def set_training_regime(self, end_to_end):
        """
        Sets whether or not to train the model end to end

        :param end_to_end: A boolean whether or not to set the training regime to end to end
        :return: None
        """

        if end_to_end:
            trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        else:
            trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="DOWN")

        self.train_op = tf.train.AdamOptimizer().minimize(self.loss, var_list=trainable_vars)

    def save_model(self, sess, save_name=None):
        """
        Saves the computation graph into a tf save format

        :param sess: The tf session to save a graph from
        :param save_name: The output file name
        :return: None
        """
        print("Saving model...")

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

        if save_name is not None:
            ckpt = tf.train.get_checkpoint_state("./checkpoints/" + save_name)
        else:
            ckpt = tf.train.get_checkpoint_state("./checkpoints/saved_model")

        if ckpt:
            print('loaded ' + ckpt.model_checkpoint_path)
            self.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('load failed')
            exit(0)
