# Nair, Siddhi
# 1001-713-489
# 2020-09-23
# Assignment-01-01

import numpy as np

class SingleLayerNN(object):
    def __init__(self, input_dimensions=2,number_of_nodes=4):
        """
        Initialize SingleLayerNN model and set all the weights and biases to random numbers.
        :param input_dimensions: The number of dimensions of the input data
        :param number_of_nodes: Note that number of neurons in the model is equal to the number of classes.
        """
        if input_dimensions == 0 or number_of_nodes == 0:
            print()
            return
        else:
            self.input_dimension = input_dimensions + 1
            self.weights = np.ones((number_of_nodes,self.input_dimension))
            self.initialize_weights()

    def initialize_weights(self,seed=None):
        """
        Initialize the weights, initalize using random numbers.
        If seed is given, then this function should
        use the seed to initialize the weights to random numbers.
        :param seed: Random number generator seed.
        :return: None
        """
        if seed != None:
            np.random.seed(seed)
        self.weights = np.random.randn(self.weights.shape[0],self.weights.shape[1])



    def set_weights(self, W):
        """
        This function sets the weight matrix (Bias is included in the weight matrix).
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
        Make a prediction on a batch of inputs.
        :param X: Array of input [input_dimensions,n_samples]
        :return: Array of model [number_of_nodes ,n_samples]
        Note that the activation function of all the nodes is hard limit.
        """
        ones = np.ones(X.shape[1])
        ones_dims = np.expand_dims(ones, axis=0)
        newX = np.insert(X, 0, ones_dims, axis=0)
        y = np.dot(self.weights,newX)
        y_hat = np.where(y<0,0,1)
        return y_hat

    def train(self, X, Y, num_epochs=10,  alpha=0.1):
        """
        Given a batch of input and desired outputs, and the necessary hyperparameters (num_epochs and alpha),
        this function adjusts the weights using Perceptron learning rule.
        Training should be repeated num_epochs times.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes ,n_samples]
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :return: None
        """
        ones = np.ones(X.shape[1])
        ones_dims = np.expand_dims(ones, axis=0)
        newX = np.insert(X, 0, ones_dims, axis=0)
        newX_t = newX.T
        for epoch in range(num_epochs):
            for idx, i in enumerate(newX_t):
                output = np.dot(self.weights, i)
                a = np.where(output < 0, 0, 1)
                e = Y.T[idx] - a
                e_dims = np.expand_dims(e, axis=0)
                i_dims = np.expand_dims(i, axis=0)
                newweight = self.get_weights() + (alpha * np.dot(e_dims.T,i_dims))
                self.set_weights(newweight)






    def calculate_percent_error(self,X, Y):
        """
        Given a batch of input and desired outputs, this function calculates percent error.
        For each input sample, if the output is not the same as the desired output, Y,
        then it is considered one error. Percent error is 100*(number_of_errors/ number_of_samples).
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes ,n_samples]
        :return percent_error
        """
        n_samples = X.shape[1]
        ones = np.ones(n_samples)
        ones_dims = np.expand_dims(ones, axis=0)
        newX = np.insert(X, 0, ones_dims, axis=0)
        y = np.dot(self.weights, newX)
        y_hat = np.where(y < 0, 0, 1)
        no_of_errors = 0

        if np.array_equal(y_hat, Y):
            no_of_errors = no_of_errors
            percent_error = 100 * (no_of_errors / n_samples)
        else:
            error = Y.T - y_hat.T
            for i in error:
                if np.array_equal(i,np.zeros(i.shape)):
                    no_of_errors = no_of_errors
                else:no_of_errors = no_of_errors + 1
            percent_error = 100 * (no_of_errors/n_samples)
        return percent_error








if __name__ == "__main__":
    input_dimensions = 2
    number_of_nodes = 2

    model = SingleLayerNN(input_dimensions=input_dimensions, number_of_nodes=number_of_nodes)
    model.initialize_weights(seed=2)
    X_train = np.array([[-1.43815556, 0.10089809, -1.25432937, 1.48410426],
                        [-1.81784194, 0.42935033, -1.2806198, 0.06527391]])
    print(model.predict(X_train))
    Y_train = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
    print("****** Model weights ******\n",model.get_weights())
    print("****** Input samples ******\n",X_train)
    print("****** Desired Output ******\n",Y_train)
    percent_error=[]
    for k in range (20):
        model.train(X_train, Y_train, num_epochs=1, alpha=0.1)
        percent_error.append(model.calculate_percent_error(X_train,Y_train))
    print("******  Percent Error ******\n",percent_error)
    print("****** Model weights ******\n",model.get_weights())