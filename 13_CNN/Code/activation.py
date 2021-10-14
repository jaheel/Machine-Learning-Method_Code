import numpy as np

class ActivationFunction(object):
    @staticmethod
    def activation(input_data, activation_name = 'relu'):
        """
        
        activation function

        Parameters
        ----------
        input_data : {array-like, tensor(3-dim)} of shape (in_data_col, in_data_row, in_data_depth)
                        if activation is softmax, the input data is vector

        activation_name : {string-like, scalar} value of {'sigmoid', 'tanh', 'relu'(default), 'softmax', 'none'}

        Returns
        -------
        input_data : {array-like, tensor(3-dim)} of shape (in_data_col, in_data_row, in_data_depth) 
                        after activation (origin data shape)

        """
        if activation_name == 'sigmoid':
            return 1.0/(1.0 + np.exp(-input_data))
        
        elif activation_name == 'tanh':
            return np.tanh(input_data)

        elif activation_name == 'relu':
            return (np.abs(input_data) + input_data)/2

        elif activation_name == 'softmax': # input data is vector
            input_data = input_data - np.max(input_data)
            exp_input_data = np.exp(input_data)
            return exp_input_data / np.sum(exp_input_data)
        
        elif activation_name == 'none': # not use any activation
            return input_data

        else:
            raise AttributeError("activation name wrong")
    
    @staticmethod
    def activation_prime(input_data, activation_name = 'relu'):
        """

        activation function derivative

        Parameters
        ----------
        input_data : {array-like, tensor(3-dim)} of shape (in_data_col, in_data_row, in_data_depth)

        activation_name : {string-like, scalar} value of {'sigmoid', 'tanh', 'relu'(default), 'softmax', 'none'}

        Returns
        -------
        result : {array-like, tensor(3-dim)} of shape (in_data_col, in_data_row, in_data_depth)
                    activation derivative
        """
        if activation_name == 'sigmoid':
            return ActivationFunction.activation(activation_name=activation_name, input_data=input_data) * (1 - ActivationFunction.activation( activation_name=activation_name, input_data=input_data))

        elif activation_name == 'tanh':
            return 1 - np.square( ActivationFunction.activation(activation_name=activation_name, input_data = input_data))
        
        elif activation_name == 'relu':
            return np.where( input_data > 0, 1, 0)
        
        elif activation_name == 'softmax':
            input_data = np.squeeze(input_data)
            length = len(input_data)
            
            result = np.zeros((length, length))

            for i in range(length):
                for j in range(length):
                    result[i, j] = ActivationFunction.__softmax(i = i, j = j, a = input_data)
            
            return result

        elif activation_name == 'none':
            return 1

        else:
            raise AttributeError("activation name wrong")
    
    @staticmethod
    def __softmax(i, j, a):
        if i == j:
            return a[i] * (1 - a[i])
        else:
            return -a[i] * a[j]

# input_data_test = np.array(
#     [
#         [
#             [1, 3, 5, 4, 7],
#             [2, 3, 2, 1, 0],
#             [7, 8, 1, 2, 3],
#             [3, 2, 9, 8, 7],
#             [2, 3, -4, 0, 2]
#         ]
#     ]
# )


# print(ActivationFunction.activation_prime(activation_name='tanh', input_data=input_data_test))