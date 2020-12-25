# Nair, Siddhi
# 1001-713-489
# 2020-10-10
# Assignment-02-01

import numpy as np


class LinearAssociator(object):
    def __init__(self, input_dimensions=2, number_of_nodes=4, transfer_function="Hard_limit"):
        """
        Initialize linear associator model
        :param input_dimensions: The number of dimensions of the input data
        :param number_of_nodes: number of neurons in the model
        :param transfer_function: Transfer function for each neuron. Possible values are:
        "Hard_limit", "Linear".
        """
        if input_dimensions == 0 or number_of_nodes == 0:
            print()
            return
        else:
            self.input_dimension = input_dimensions
            self.weights = np.ones((number_of_nodes, self.input_dimension))
            self.transfer_function = transfer_function.lower()
            self.initialize_weights()



    def initialize_weights(self, seed=None):
        """
        Initialize the weights, initalize using random numbers.
        If seed is given, then this function should
        use the seed to initialize the weights to random numbers.
        :param seed: Random number generator seed.
        :return: None
        """
        if seed != None:
            np.random.seed(seed)
        self.weights = np.random.randn(self.weights.shape[0], self.weights.shape[1])

    def set_weights(self, W):
        """
         This function sets the weight matrix.
         :param W: weight matrix
         :return: None if the input matrix, w, has the correct shape.
         If the weight matrix does not have the correct shape, this function
         should not change the weight matrix and it should return -1.
         """
        if W.shape[0] == self.weights.shape[0] and W.shape[1] == self.weights.shape[1]:
            self.weights = W
            return None
        else:
            return -1

    def get_weights(self):
        """
         This function should return the weight matrix(Bias is included in the weight matrix).
         :return: Weight matrix
         """
        return self.weights

    def predict(self, X):
        """
        Make a prediction on an array of inputs
        :param X: Array of input [input_dimensions,n_samples].
        :return: Array of model outputs [number_of_nodes ,n_samples]. This array is a numerical array.
        """
        output = np.dot(self.weights, X)
        if self.transfer_function == "hard_limit":
            model_op = np.where(output < 0, 0, 1)
        else:
            model_op = output
        return model_op




    def fit_pseudo_inverse(self, X, y):
        """
        Given a batch of data, and the targets,
        this function adjusts the weights using pseudo-inverse rule.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        """

        #using numpy funciton pinv() to get the pseudo inverse
        psuedo_inv = np.linalg.pinv(X)

        self.set_weights(np.dot(y, psuedo_inv))

    def train(self, X, y, batch_size=5, num_epochs=10, alpha=0.1, gamma=0.9, learning="Delta"):
        """
        Given a batch of data, and the necessary hyperparameters,
        this function adjusts the weights using the learning rule.
        Training should be repeated num_epochs time.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples].
        :param num_epochs: Number of times training should be repeated over all input data
        :param batch_size: number of samples in a batch
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :param gamma: Controls the decay
        :param learning: Learning rule. Possible methods are: "Filtered", "Delta", "Unsupervised_hebb"
        :return: None
        """
        learning = learning.lower()

        for i in range(num_epochs):
            for x in range(0, np.shape(X)[1], batch_size):
                col = x + batch_size
                batchX = X[:, x:col]
                batchy = y[:, x:col]


                a = self.predict(batchX)
                e = batchy - a

                if learning == 'delta':
                    delta = alpha * np.dot(e, np.transpose(batchX))
                    self.weights = self.weights + delta
                elif learning == 'filtered':
                    learn_filter = alpha * np.dot(batchy, np.transpose(batchX))
                    self.weights = (1 - gamma) * self.weights + learn_filter
                elif learning == 'unsupervised_hebb':
                    unsuper = alpha * np.dot(a, np.transpose(batchX))
                    self.weights = self.weights + unsuper



    def calculate_mean_squared_error(self, X, y):
        """
        Given a batch of data, and the targets,
        this function calculates the mean squared error (MSE).
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        :return mean_squared_error
        """
        error_square = np.square(y - self.predict(X))
        sum =np.sum(error_square)
        xshape = np.shape(X)
        MSE = sum / (xshape[1] * xshape[0])

        return MSE