import numpy as np
from Layer import Layer
from PoolMethod import PoolMethod

class PoolLayer(Layer):
    def __init__(self, input_shape, filter_size = (2, 2), strides = (2, 2), pool_method = 'MAX', name = 'none'):
        """Pool layer init

        Parameters
        ----------
        input_shape : {tuple-like, vector_3} of shape (in_data_col, in_data_row, in_data_channel)

        filter_size : {tuple-like, vector_2} of shape (filter_col, filter_row) default is (2, 2)

        strides : {tuple-like, vector_2} of shape (col_stride, row_stride) default is (2, 2)

        pool_method : {string-like, scalar} value of {'MAX'(default), 'AVG'}

        name : {string-like, scalar} default is 'none'
        """
        
        self.input_shape = input_shape
        self.filter_size = filter_size
        self.pool_method = pool_method
        self.name = name
        self.strides = strides

    def forward_propagation(self, in_data):
        """forward

        Parameters
        ----------
        in_data : {array-like, tensor(3-dim)} of shape (in_data_col, in_data_row, in_data_channel)

        Returns
        -------
        result : {array-like, tensor(3-dim)} of shape (after_pool_out_data_col, after_pool_out_data_row, in_data_channel)

        """
        self.__input = in_data
        train_pool_method = PoolMethod(in_data = in_data, filter_size = self.filter_size, strides = self.strides, pool_name = self.pool_method)
        self.__output = train_pool_method.forward_pass()

        return self.__output

    def back_propagation(self, error, learn_rate = 1):
        """backward

        Parameters
        ----------
        error : {array-like, tensor(3-dim)} of shape (out_data_col, out_data_row, channel)

        learn_rate : {float-like, scalar}

        Returns
        -------
        error_pre : {array-like, tensor(3-dim)} of shape (in_data_col, in_data_row, channel)

        """
        train_pool_method = PoolMethod(in_data = self.__input, filter_size = self.filter_size, strides = self.strides, pool_name = self.pool_method)
        self.__error = train_pool_method.backward_pass(error = error, out_data = self.__output)

        return self.__error

# --------------------
# test : forward and back
# test_data = np.random.randint(10, size=(5, 5, 1))
# test_error = np.random.randint(low = 1, high = 4, size=(2, 2, 1))
# print(test_data)
# print("---------------------")
# print(test_error)
# test_pool_layer = PoolLayer(input_shape = test_data.shape, pool_method = 'AVG')
# # test_pool_layer = PoolLayer(input_shape = (5, 5, 3), filter_size = (3, 3), strides = (2, 2), pool_method = 'MAX')
# print("---------------------")
# print(test_pool_layer.forward_propagation(in_data = test_data))
# print("---------------------")
# print(test_pool_layer.back_propagation(error = test_error))