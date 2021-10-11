import numpy as np

class NNActivator(object):
    def __init__(self):
        pass

    def __sigmoid(self, data):
        """
        
        Sigmoid Function : 1/(1 + exp(-x))

        Parameters
        ----------
        data : {array-like, sparse matrix} of shape(1_samples, n_features)
        
        Returns
        -------
        result : {array-like, python_list } of shape(1_samples, n_activate_result)

        """
    
        result = []

        for element in data:
            result.append( 1/(1+np.exp(-element))) 
        return result

    def __tanh(self, data):
        """
        
        tanh Function : \\frac{exp(x) - exp(-x)}{exp(x) + exp(-x)}

        Parameters
        ----------
        data : {array-like, sparse matrix} of shape(1_samples, n_features)
        
        Returns
        -------
        result : {array-like, python_list } of shape(1_samples, n_activate_result)
        
        """
        
        result = []
        
        for element in data:
            pos_e = np.exp(element)
            neg_e = np.exp(-element)
            result.append((pos_e - neg_e)/(pos_e + neg_e))
        
        return result
    
    def __relu(self, data):
        """
        
        ReLU Function : f(x) = max(0, x)

        Parameters
        ----------
        data : {array-like, sparse matrix} of shape(1_samples, n_features)
        
        Returns
        -------
        result : {array-like, python_list } of shape(1_samples, n_activate_result)
        
        """
        
        result = np.maximum(0, data)
        return result
    
    def __leaky_relu(self, data):
        """
        
        Leaky ReLU Function : f(x) = max(0.01x, x)

        Parameters
        ----------
        data : {array-like, sparse matrix} of shape(1_samples, n_features)
        
        Returns
        -------
        result : {array-like, python_list } of shape(1_samples, n_activate_result)
        
        """
      
        result = np.maximum(0.01 * data, data)
        return result

    def __exponential_linear_units(self, data, alpha = 1.0):
        """
        
        ELU Function : f(x) = x (if x>0) or f(x) = alpha*(exp(x)-1) (otherwise)

        Parameters
        ----------
        data : {array-like, sparse matrix} of shape(1_samples, n_features)
        
        Returns
        -------
        result : {array-like, python_list } of shape(1_samples, n_activate_result)
        
        """

        result = []

        for element in data:
            if element > 0:
                result.append(element)
            else:
                result.append(alpha * (np.exp(element)-1) )
        
        return result

    def __softmax(self, data):
        """
        
        Softmax Function : f(x) = exp(i)/ (\\sum_j(exp(j)))

        Parameters
        ----------
        data : {array-like, sparse matrix} of shape(1_samples, n_features)
        
        Returns
        -------
        result : {array-like, python_list } of shape(1_samples, n_activate_result)
        
        """
        exp_data = np.exp(data)
        exp_data_sum = np.sum(exp_data)
        result = exp_data/exp_data_sum
        
        return result

    def fit(self, data, function_name="relu", alpha = 1.0):
        """
        
        activate data

        Parameters
        ----------
        data : {array-like, sparse matrix} of shape(1_samples, n_features)

        function_name : 
            1. relu
            2. sigmoid
            3. tanh
            4. leaky_relu
            5. elu

        alpha : if function is elu, use this variable(default: 1.0)
        
        Returns
        -------
        result : {array-like, python_list } of shape(1_samples, n_activate_result)
        
        """
        data = np.array(data)
        if function_name == "relu":
            result = self.__relu(data=data)
        elif function_name == "sigmoid":
            result = self.__sigmoid(data=data)
        elif function_name == "tanh":
            result = self.__tanh(data=data)
        elif function_name == "leaky_relu":
            result = self.__leaky_relu(data=data)
        elif function_name == "elu":
            result = self.__exponential_linear_units(data=data, alpha=alpha)
        else:
            pass

        return result

    def softmax_fit(self, data):
        """
        
        softmax data

        Parameters
        ----------
        data : {array-like, sparse matrix} of shape(1_samples, n_features)
        
        Returns
        -------
        result : {array-like, python_list } of shape(1_samples, n_activate_result)
        
        """
        data = np.array(data)
        result = self.__softmax(data)

        return result




