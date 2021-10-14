import numpy as np

class Padding2D:
    @staticmethod
    def padding_matrix(input_data, filter_size, pad_add_number, padding = "SAME"):
        """padding matrix data
        
        Padding the specific input matrix with the specific padding pattern

        Parameters
        ----------
        input_data : {array-like, matrix} of shape (in_data_col, in_data_row)

        filter_size : {tuple-like, vector_2} of shape (filter_height, filter_wide)

        pad_add_number : {tuple-like, vector_2} of shape (p_col_add, p_row_add)
        
        padding : {string-like, scalar} default is 'SAME' (optional: 'SAME', 'VALID')

        Returns
        -------
        input_data : {array-like, matrix} input data after padding

        """

        if padding =='VALID':
            return input_data

        row = np.zeros(input_data.shape[0])
        for row_index in range(pad_add_number[1]):
            input_data = np.insert(input_data, 0, values = row, axis = 1)
            input_data = np.insert(input_data, input_data.shape[1], values = row, axis = 1)
        
        col = np.zeros(input_data.shape[1])
        for col_index in range(pad_add_number[0]):
            input_data = np.insert(input_data, 0, values = col, axis = 0)
            input_data = np.insert(input_data, input_data.shape[0], values = col, axis = 0)

        return input_data

    @staticmethod
    def padding_tensor(input_data, filter_size, strides = (1,1), padding = "SAME"):
        """padding tensor(3-dim) data
        
        padding origin data of multi-channel

        Parameters
        ----------
        input_data : {array-like, tensor(3-dim)} of shape (in_data_col, in_data_row, channel)

        filter_size : {tuple-like, vector_2} of shape (filter_col, filter_row)

        strides : {tuple-like, vector_2} of shape (col_stride, row_stride)

        padding : {string-like, scalar} default is 'SAME' (optional: 'SAME', 'VALID')

        Returns
        -------
        result : {array-like, tensor(3-dim)} input data after padding

        """
        if padding =='VALID':
            return input_data
        
        out_data_height = int(np.ceil(input_data.shape[0] / strides[0]))
        out_data_wide = int(np.ceil(input_data.shape[1] / strides[1]))

        p_height = np.max( (out_data_height - 1) * strides[0] + filter_size[0] - input_data.shape[0], 0)
        p_wide = np.max( (out_data_wide - 1) * strides[1] + filter_size[1] - input_data.shape[1], 0)

        p_col_add = int(np.floor(p_height / 2))
        p_row_add = int(np.floor(p_wide / 2))

        result = np.zeros( shape=(input_data.shape[0] + 2 * p_col_add, input_data.shape[1] + 2 * p_row_add, input_data.shape[2]))   
        for channel_index in range(input_data.shape[2]):
            result[:, :, channel_index] = Padding2D.padding_matrix(input_data = input_data[:, :, channel_index], filter_size = filter_size,pad_add_number = (p_col_add, p_row_add), padding = padding)

        return result


# input_data_test = np.random.randint(5, size=(5, 5, 2))
# print(input_data_test)
# print(Padding2D.padding_tensor(input_data=input_data_test, filter_size=(3,3), strides=(2,2), padding="SAME"))

# cov_filter = np.array(
#     [
#         [0, 0, 0],
#         [0, 1, 0],
#         [0, 0, 0]
#     ]
# )