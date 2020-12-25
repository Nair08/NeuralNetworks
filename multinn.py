# Nair, Siddhi
# 1001-713-489
# 2020_10_25
# Assignment-03-01

# %tensorflow_version 2.x
import tensorflow as tf
import numpy as np


class MultiNN(object):
    def __init__(self, input_dimension):
        """
        Initialize multi-layer neural network
        :param input_dimension: The number of dimensions for each input data sample
        """
        self.input_dimensions = input_dimension
        self.weights = []
        self.bias = []

        self.transfer_func = []
        self.layer_dim = []
        self.layers = 0


    def add_layer(self, num_nodes, transfer_function="Linear"):
        """
         This function adds a dense layer to the neural network
         :param num_nodes: number of nodes in the layer
         :param transfer_function: Activation function for the layer. Possible values are:
        "Linear", "Relu","Sigmoid".
         :return: None
         """
        self.num_nodes = num_nodes
        self.layer_dim.append(num_nodes)
        transfer_function = transfer_function.lower()

        if self.layers == 0:
            self.transfer_func.append(transfer_function)
            self.weights.append(np.random.randn(self.input_dimensions, self.num_nodes))
            self.bias.append(np.random.randn(self.num_nodes, 1))

        else:
            self.transfer_func.append(transfer_function)
            self.weights.append(np.random.randn(self.layer_dim[self.layers-1], self.num_nodes))
            self.bias.append(np.random.randn(self.num_nodes, 1))

        self.layers += 1



    def get_weights_without_biases(self, layer_number):
        """
        This function should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero.
         :param layer_number: Layer number starting from layer 0. This means that the first layer with
          activation function is layer zero
         :return: Weight matrix for the given layer (not including the biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         """

        return self.weights[layer_number]

    def get_biases(self, layer_number):
        """
        This function should return the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0
         :return: Weight matrix for the given layer (not including the biases).
         Note that the biases shape should be [1][number_of_nodes]
         """

        return self.bias[layer_number]

    def set_weights_without_biases(self, weights, layer_number):
        """
        This function sets the weight matrix for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0
         :return: none
         """

        self.weights[layer_number] = weights


    def set_biases(self, biases, layer_number):
        """
        This function sets the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0
        :return: none
        """

        self.bias[layer_number] = biases




    def calculate_loss(self, y, y_hat):
        """
        This function calculates the sparse softmax cross entropy loss.
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
        :param y_hat: Array of actual output values [n_samples][number_of_classes].
        :return: loss
        """
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_hat))
        return loss

    def predict(self, X):
        """
        Given array of inputs, this function calculates the output of the multi-layer network.
        :param X: Array of input [n_samples,input_dimensions].
        :return: Array of outputs [n_samples,number_of_classes ]. This array is a numerical array.
        """
        tf_x = tf.Variable(X)
        for (weight, b, trfunc) in zip(self.weights, self.bias, self.transfer_func):
            z = tf.add(tf.matmul(tf_x, weight), tf.transpose(b))

            if trfunc == 'sigmoid':
                tf_x = tf.math.sigmoid(z)

            elif trfunc == 'relu':
                tf_x = tf.nn.relu(z)

            elif trfunc == 'linear':
                tf_x = z
        return tf_x


    def train(self, X_train, y_train, batch_size, num_epochs, alpha=0.8):
        """
         Given a batch of data, and the necessary hyperparameters,
         this function trains the neural network by adjusting the weights and biases of all the layers.
         :param X: Array of input [n_samples,input_dimensions]
         :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :param alpha: Learning rate
         :return: None
         """
        X_train = tf.Variable(X_train)
        y_train = tf.Variable(y_train)
        samples = np.shape(X_train)[0]

        for i in range(num_epochs):
            for j in range(0, samples, batch_size):
                row = j + batch_size
                batch_x = X_train[j:row, :]
                batch_y = y_train[j:row]

                with tf.GradientTape() as gt:
                    predictions = self.predict(batch_x)
                    loss = self.calculate_loss(batch_y, predictions)
                    dloss_dw, dloss_db = gt.gradient(loss, [self.weights, self.bias])

                for k in range(self.layers):
                    self.weights[k].assign_sub(alpha * dloss_dw[k])
                    self.bias[k].assign_sub(alpha * dloss_db[k])


    def calculate_percent_error(self, X, y):
        """
        Given input samples and corresponding desired (true) output as indexes,
        this method calculates the percent error.
        For each input sample, if the predicted class output is not the same as the desired class,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        Note that the predicted class is the index of the node with maximum output.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return percent_error
        """
        x = self.predict(X)
        error = y - np.argmax(x, axis=1)

        no_of_samples = np.shape(y)[0]
        not_zeroes = np.count_nonzero(error)

        percent_error = not_zeroes / no_of_samples

        return percent_error

    def calculate_confusion_matrix(self, X, y):
        """
        Given input samples and corresponding desired (true) outputs as indexes,
        this method calculates the confusion matrix.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return confusion_matrix[number_of_classes,number_of_classes].
        Confusion matrix should be shown as the number of times that
        an image of class n is classified as class m.
        """
        x = self.predict(X)

        y_hat = np.argmax(x, axis=1)
        confusion_matrix = tf.math.confusion_matrix(y, y_hat)

        return confusion_matrix