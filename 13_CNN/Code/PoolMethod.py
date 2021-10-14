import numpy as np

class PoolMethod(object):
    def __init__(self, in_data, filter_size = (2,2), strides = (1,1), pool_name ='MAX'):
        """tensor pool
        

        Parameters
        ----------
        in_data : {array-like, tensor(3-dim)} of shape (in_data_col, in_data_row, in_data_channel) 
        
        filter_size : {tuple-like, vector_2} of shape (filter_col, filter_row)
        
        strides : {tuple-like, vector_2} of shape (col_stride, row_stride)

        pool_name : {string-like, scalar}  value of {'MAX'(default), 'AVG'}
        """
        self.in_data = in_data
        self.filter_size = filter_size
        self.strides = strides
        self.pool_name = pool_name
        self.channel = in_data.shape[2]

        self.output_shape = ( int((in_data.shape[0] - filter_size[0]) / strides[0]) + 1, 
                              int((in_data.shape[1] - filter_size[1]) / strides[1]) + 1,
                              self.channel)

    def __max_pool_single(self, in_data):
        """single max pool scalar



        Parameters
        ----------
        in_data : {array-like, matrix} of shape(filter_col, filter_row) 
        
        Returns
        -------
        result : {float-like, scalar} the max result
        """
        if in_data.shape[0] != self.filter_size[0] or in_data.shape[1] != self.filter_size[1]:
            return None
        
        result = 0
        
        for x_index in range(self.filter_size[0]):
            for y_index in range(self.filter_size[1]):
                result = np.maximum(result, in_data[x_index][y_index])
            
        return result
    
    def __average_pool_single(self, in_data):
        """single average pool scalar

        

        Parameters
        ----------
        in_data : {array-like, matrix} of shape(filter_col, filter_row) 
        
        Returns
        -------
        result : {float-like, scalar} the average result
        """
        if in_data.shape[0] != self.filter_size[0] or in_data.shape[1] != self.filter_size[1]:
            return None
        
        result = 0
        
        for x_index in range(self.filter_size[0]):
            for y_index in range(self.filter_size[1]):
                result += in_data[x_index][y_index]
            
        return result/(self.filter_size[0] * self.filter_size[1])
        
    
    def __max_pool(self, in_data):
        """max pool matrix

        C : side_length

        C_output = (C_input - C_kernel)/strides + 1

        Parameters
        ----------
        in_data : {array-like, matrix} of shape (in_data_col, in_data_row)
        
        Returns
        -------
        result : {array-like, matrix} of shape( (in_data_col - filter_col) / stride_col + 1, (in_data_row - filter_row) / stride_row + 1)
        """
        
        result = []
        filter_number_h = self.filter_size[0] - 1
        filter_number_w = self.filter_size[1] - 1

        for y_index in range(0, in_data.shape[0] - filter_number_h, self.strides[0]):
            result_x = []

            for x_index in range(0, in_data.shape[1] - filter_number_w, self.strides[1]):
                max_pool_in_data = in_data[y_index : y_index + self.filter_size[0], x_index : x_index + self.filter_size[1]]
                result_x.append(self.__max_pool_single(in_data = max_pool_in_data))
            
            result.append(result_x)
        
        return np.array(result)

    def __average_pool(self, in_data):
        """average pool matrix

        C : side_length

        C_output = (C_input - C_kernel)/strides + 1

        Parameters
        ----------
        in_data : {array-like, matrix} of shape (in_data_col, in_data_row)
        
        Returns
        -------
        result : {array-like, matrix} of shape( (in_data_col - filter_col) / stride_col + 1, (in_data_row - filter_row) / stride_row + 1)
        """
        
        result = []
        filter_number_h = self.filter_size[0] - 1
        filter_number_w = self.filter_size[1] - 1

        for y_index in range(0, in_data.shape[0] - filter_number_h, self.strides[0]):
            result_x = []

            for x_index in range(0, in_data.shape[1] - filter_number_w, self.strides[1]):
                average_pool_in_data = in_data[y_index : y_index + self.filter_size[0], x_index : x_index + self.filter_size[1]]
                result_x.append(self.__average_pool_single(in_data = average_pool_in_data))
            
            result.append(result_x)
        
        return np.array(result)

    def __find_max_index(self, in_data, find_scalar):
        """find max index

        Parameters
        ----------
        in_data : {array-like, matrix} of shape (filter_col, filter_row)

        find_scalar : {float-like, scalar}

        Returns
        -------
        scalar_index : {array-like, vector_2} of value [h_index, w_index]
        """
        for h_index in range(in_data.shape[0]):
            for w_index in range(in_data.shape[1]):
                if in_data[h_index][w_index] == find_scalar:
                    return np.array([h_index, w_index]) 
        

    def forward_pass(self):
        """pool method forward


        Returns
        -------
        result : {array-like, tensor(3-dim)} of shape (out_data_col, out_data_row, channel)
        """

        result = np.zeros(self.output_shape)

        if self.pool_name == 'MAX':
            for channel_index in range(self.channel):
                result[:, :, channel_index] = self.__max_pool(in_data = self.in_data[:, :, channel_index])

        elif self.pool_name == 'AVG':
            for channel_index in range(self.channel):
                result[:, :, channel_index] = self.__average_pool(in_data = self.in_data[:, :, channel_index])

        else:
            ValueError("pool method name is wrong")

        return result

    def backward_pass(self, error, out_data):
        """pool method backward

        Parameters
        ----------
        error : {array-like, tensor(3-dim)} of shape (out_data_col, out_data_row, channel)

        out_data : {array-like, tensor(3-dim)} of shape (out_data_col, out_data_row, channel)

        Returns
        -------
        error_pre : {array-like, tensor(3-dim)} of shape (in_data_col, in_data_row, channel)
                    previous layer error tensor
        """
        result = np.zeros(self.in_data.shape)
        
        if self.pool_name == 'MAX':
            for channel_index in range(self.channel):
                for o_h_index in range(out_data.shape[0]):
                    for o_w_index in range(out_data.shape[1]):
                        in_h_index = o_h_index * self.strides[0]
                        in_w_index = o_w_index * self.strides[1]
                        scalar_find_index = self.__find_max_index(in_data = self.in_data[in_h_index : in_h_index + self.filter_size[0], in_w_index : in_w_index + self.filter_size[1], channel_index], find_scalar = out_data[o_h_index, o_w_index, channel_index])
                        result[in_h_index + scalar_find_index[0], in_w_index + scalar_find_index[1], channel_index] = error[o_h_index, o_w_index, channel_index]

        elif self.pool_name == 'AVG':
            for channel_index in range(self.channel):
                for o_h_index in range(out_data.shape[0]):
                    for o_w_index in range(out_data.shape[1]):
                        in_h_index = o_h_index * self.strides[0]
                        in_w_index = o_w_index * self.strides[1]
                        result[in_h_index : in_h_index + self.filter_size[0], in_w_index : in_w_index + self.filter_size[1], channel_index] = error[o_h_index, o_w_index, channel_index] / (self.filter_size[0] * self.filter_size[1])
        
        else:
            ValueError("pool method name is wrong")
        
        return result




    

# ------------
# test : max_pool_single

# test_data = np.array(
#     [
#         [1, 2],
#         [4, 6]
#     ]
# )

# testPool = MaxPool(test_data)
# print(testPool.max_pool_single(in_data = test_data))
# -------------------

# ----------------
# test : max_pool
# test_data = np.array(
#     [
#         [1, 2, 5, 7, 5],
#         [4, 6, 7, 8, 4],
#         [0, 4, 6, 10, 6],
#         [0, 3, 6, 3, 7],
#         [5, 8, 4, 3, 2]
#     ]
# )

# test_data_2 = np.array(
#     [
#         [1, 2, 5, 7],
#         [4, 6, 7, 8],
#         [0, 4, 6, 10],
#         [0, 3, 6, 3]
#     ]
# )

# test_data = np.random.randint(10, size=(5, 5, 3))
# print(test_data)
# test_max_pool = PoolMethod(in_data = test_data, filter_size = (2, 2), strides = (2, 2), pool_name = 'AVG')
# print(test_max_pool.fit())
# -------------------

# --------------
# test : average pool

# test_data = np.array(
#     [
#         [1, 2, 5, 7, 5],
#         [4, 6, 7, 8, 4],
#         [0, 4, 6, 10, 6],
#         [0, 3, 6, 3, 7],
#         [5, 8, 4, 3, 2]
#     ]
# )

# test_data_2 = np.array(
#     [
#         [1, 2, 5, 7],
#         [4, 6, 7, 8],
#         [0, 4, 6, 10],
#         [0, 3, 6, 3]
#     ]
# )

# test_average_pool = PoolMethod(in_data=test_data_2, filter_size=2, stride=1)
# print(test_average_pool.average_pool())

# --------------