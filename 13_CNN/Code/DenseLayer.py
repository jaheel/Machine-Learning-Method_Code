import numpy as np
from Layer import Layer
from activation import ActivationFunction

class DenseLayer(Layer):
    def __init__(self, shape, activation = 'relu', name = 'none'):
        """dense layer init

        Parameters
        ----------
        shape : {tuple-like, vector_2} of shape (in_neurals, out_neurals)

        activation : {string-like, scalar} value of {'sigmoid', 'tanh', 'relu'(default), 'softmax','none'}

        name : {string-like, scalar} current layer name

        """
        self.shape = shape
        self.activation_name = activation
        self.name = name
        self.__weights = np.random.randn(shape[0], shape[1])
        self.__bias = np.random.randn(1, shape[1])

    def __update_params(self, weights_delta, bias_delta, learn_rate):
        """update parameters

        Parameters
        ----------
        weights_delta : {array-like, matrix} of shape (in_neurals, out_neurals)

        bias_delta : {array-like, matrix} of shape (1, out_neurals)

        learn_rate : {float-like, scalar} 
        """
        
        self.__weights -= weights_delta * learn_rate
        self.__bias -= bias_delta * learn_rate
        

    def forward_propagation(self, in_data):
        """forward

        Parameters
        ----------
        in_data : {array-like, vector} of shape (in_neurals)
        
        Returns
        -------
        result : {array-like, vector} of shape (out_neurals)
        """
        self.__input = np.array(in_data)
        
        self.__output = ActivationFunction.activation(input_data = self.__input.dot(self.__weights) + self.__bias, activation_name = self.activation_name)

        return self.__output

    
    def back_propagation(self, error, learn_rate):
        """backward

        Parameters
        ----------
        error : {array-like, vector} of shape (out_neurals)

        learn_rate : {float-like, scalar}

        Returns
        -------
        error_pre : {array-like, vector} of shape (in_neurals)
        """
        out_delta = error * ActivationFunction.activation_prime(input_data = self.__output, activation_name = self.activation_name)
        
        if self.activation_name == 'softmax':
            out_delta = error.dot(ActivationFunction.activation_prime(input_data = self.__output, activation_name = self.activation_name))
            out_delta = np.matrix(out_delta)

        weights_delta = np.matrix(self.__input).T.dot(out_delta)

        input_delta = out_delta.dot(self.__weights.T)

        self.__update_params(weights_delta = weights_delta, bias_delta = out_delta, learn_rate = learn_rate)

        result = np.array(input_delta)
        return result.flatten()


# test_data = np.random.randint(5, size=(5))
# print(test_data)

# test_layer = DenseLayer(shape = (5, 3), activation = 'softmax')
# test_out = test_layer.forward_propagation(in_data = test_data)
# print(test_out)

# test_error = np.array([1, 0, 1])
# print(test_error.shape)
# test_pre_error = test_layer.back_propagation( error = test_error, learn_rate = 1)
# print(test_pre_error)