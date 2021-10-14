import numpy as np
from Layer import Layer

class FlattenLayer(Layer):
    def __init__(self):
        self.__input_shape = None
        self.name = 'flatten'
    
    def forward_propagation(self, in_data):
        """forward

        Parameters
        ----------
        in_data : {array-like, tensor(3-dim)} of shape (in_data_col, in_data_row, in_data_channel)
        
        Returns
        -------
        result : {array-like, vector} flatten the data
        """
        self.__input_shape = in_data.shape
        return in_data.flatten()

    def back_propagation(self, error, learn_rate = 1):
        """backward

        Parameters
        ----------
        error : {array-like, vector_out_neurals} 
        
        Returns
        -------
        error_pre : {array-like, tensor(3-dim)} of shape (in_data_col, in_data_row, in_data_channel)
        """
        
        return np.resize(error, self.__input_shape)
        

# test_data = np.random.randint(5, size=(1, 1, 20))
# test_model = FlattenLayer()
# print(test_model.forward_propagation(in_data = test_data))
# test_error = np.random.randint(5, size=(120))
# print(test_model.back_propagation(error=test_error))
