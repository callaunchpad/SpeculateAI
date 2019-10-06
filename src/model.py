import tensorflow as tf

"""
This is the basic structure of the models in this project
"""

class Model:
    def __init__(self, hyper_parameters):
        """
        Sets up any hyper-parameters that we may need for the model and builds the graph
        """
        self.build_graph(hyper_parameters)

    def build_graph(self, hyper_parameters):
        """
        Builds the computation graph this model will be working with.

        Will build the graph with specific fields:
            - self.inputs: an input tf placeholder that will be called
            - self.output: a tensorflow operation to be evaluated for network output
            - self.loss: a tensorflow operation that evaluates to the loss given inputs and outputs
            - self.train_op: a training operation we will call during training

        :return: None
        """

        self.inputs = None
        self.output = None
        self.loss = None
        self.train_op = None

        return

    def train_step(self, inputs, outputs, sess):
        """
        Performs a training step using self.train_op in the passed session.

        :param inputs: The inputs to forward propagate
        :param outputs: The outputs used in computing the loss function
        :param sess: The tensorflow session used for evaluating the training operation
        :return: The loss on <inputs> and <outputs>
        """

        feed_dict = {
            self.inputs: inputs,
            self.outputs: outputs
        }

        loss_value, _ = sess.run([self.loss, self.train_op], feed_dict=feed_dict)

        return loss_value

    def get_loss(self, inputs, outputs, sess):
        """
        Evaluates the loss on the given inputs and outputs in the given TF session

        :param inputs: The inputs to forward propagate
        :param outputs: The outputs used in computing the loss function
        :param sess: The tensorflow session used for evaluating the loss operation
        :return: The loss on <inputs> and <outputs>
        """
        
        feed_dict = {
            self.inputs: inputs,
            self.output: outputs
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
