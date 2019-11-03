import tensorflow as tf

"""
This is the basic structure of the models in this project.
In this context, model is a substructure of the overall aggregated graph
"""

class BaseModel:
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
            self.inputs = None
            self.output = None
            self.labels = None
            self.loss = None
            self.train_op = None

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

    def save_model(self, save_name=None):
        """
        Saves the model in the checkpoints folder
        :param save_name: The name under which to save the model
        :return: None
        """
        print("Saving model...")
        if save_name is not None:
            self.saver.save(self.sess, "./checkpoints/UNet" + save_name)
            return

        self.saver.save(self.sess, "./checkpoints/UNet" + str(self.start_channel_depth))

    def load_model(self, starting_depth):
        """
        Loads in the pre-trained weights from the specified model
        :param starting_depth: Specifies a model to load by the starting channel depth
        :return: None
        """
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        ckpt = tf.train.get_checkpoint_state("./checkpoints/")
        if ckpt:
            print('loaded ' + ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print('load failed')
            exit(0)
