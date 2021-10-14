import numpy as np
from activation import ActivationFunction


class cov2D(object):
    def __init__(self, input_data, filter, strides = (1,1)):
        """

        Parameters
        ----------
        input_data : {array-like, matrix} of shape (n_col, n_row) origin data, 2D matrix
        
        filter : {array-like, matrix} of shape (filter_size, filter_size) filter matrix (3,3) or (5,5)
        
        strides : {tuple-like, vector_2} of shape (col_stride, row_stride) default is (1,1)
        
        """
        self.input_data = input_data
        self.filter = filter
        self.strides = strides

    def __cov_single(self, input_data):
        """single convolution calculation
    
        The input data shape is same to filter shape. The result is calculated 
        with the specific single input scalar and the specific single filter 
        scalar one by one.

        Paramters
        ---------
        input_data : {array-like, matrix} of shape(filter_size, filter_size)

        Returns
        -------
        result : {float-like, scalar}

        """
        if input_data.shape[0] != self.filter.shape[0] or input_data.shape[1] != self.filter.shape[1]:
            pass
        
        result = 0

        for shape_0_index in range(input_data.shape[0]):
            for shape_1_index in range(input_data.shape[1]):
                result += input_data[shape_0_index][shape_1_index] * self.filter[shape_0_index][shape_1_index]

        return result
    
    def convolution(self):
        """A convolution operation of input matrix

        The input data is a matrix(2-dim), The specific matrix convolution with 
        the specific filter. The result is the matrix after convolution. 
    
        Returns
        -------
        result : {array-like, matrix} of shape ( (input_size-filter_size)/stride + 1, (input_size-filter_size)/stride + 1 )
        """


        filter_number = int((self.filter.shape[0] - 1)/2)

        result = []

        for y_index in range(0, self.input_data.shape[0] - self.filter.shape[0] + 1, self.strides[0]):
            result_x = []
            
            for x_index in range(0, self.input_data.shape[1]-self.filter.shape[1] + 1, self.strides[1]):
                input_matrix = self.input_data[y_index : y_index + self.filter.shape[0], x_index : x_index + self.filter.shape[1]]
                result_x.append(self.__cov_single(input_data = input_matrix))

            result.append(result_x)

        return np.array(result)


class ConvKernel(object):
    def __init__(self, kernel_size, input_shape, strides):
        """convolution kernel tensor(3-dim) init
        
        convolution kernel init with the specific number 
        of channels that same to input data channel

        Parameters
        ----------
        kernel_size : {tuple-like, vector_2} of shape (kernel_col, kernel_row)

        input_shape : {tuple-like, vector_3} of shape (in_data_col, in_data_row, channel)

        strides : {tuple-like, vector_2} of shape (col_stride, row_stride)
        """
        self.__kernel_height = kernel_size[0]
        self.__kernel_wide = kernel_size[1]
        self.__input_shape = input_shape
        self.__channel = input_shape[2]
        self.__strides = strides

        self.__weights = np.random.randn(kernel_size[0], kernel_size[1], input_shape[2])
        
        self.__output_shape = ( int((input_shape[0] - kernel_size[0]) / strides[0]) + 1, 
                                int((input_shape[1] - kernel_size[1]) / strides[1]) + 1)
        
        self.__input = None
        self.__output = None
        self.__b = 0 # np.random.randn(self.__output_shape[0], self.__output_shape[1])
    
    def kernel_property_print(self):
        """print instance property

        print all the property of this instance that
        include : kernel shape, input shape, strides,
        weights, output shape, bias, in_data, out_data

        """
        print("kernel height : ", self.__kernel_height)
        print("kernel wide : ", self.__kernel_wide)
        print("input shape : ", self.__input_shape)
        print("channel : ", self.__channel)
        print("strides : ", self.__strides)
        print("weights : ", self.__weights)
        print("output shape : ", self.__output_shape)
        print("input : ", self.__input)
        print("output : ", self.__output)
        print("b : ", self.__b)

    def __flip_weights(self):
        """rotate weights tensor 180°
        
        After testing, multi-channel can also operate normally

        Returns
        -------
        result : {array-like, tensor(3-dim)} of shape (w_col, w_row, w_channel)
        """
 
        return np.fliplr(np.flipud(self.__weights))

    def __update_params(self, w_delta, b_delta, learn_rate):
        """update parameters(W, b)
        
        classic gradient descent 

        Parameters
        ----------
        w_delta : {array-like, tensor(3-dim)} of shape (kernel_col, kernel_row, channel)

        b_delta : {float-like, scalar} 

        learn_rate : {float-like, scalar}
        """
        self.__weights -= w_delta * learn_rate
        self.__b -= b_delta * learn_rate

    def __single_conv(self, input_data, filter, strides):
        """single convolution of input matrix with filter matrix
        
        'One input matrix' conv 'one filter matrix' to 'one feature map' 

        Parameters
        ----------
        input_data : {array-like, matrix} of shape (in_data_col, in_data_row) 

        weights : {array-like, matrix} of shape (filter_col, filter_row) 

        strides : {tuple-like, vector_2} of shape (col_stride, row_stride)

        Returns
        -------
        result : {array-like, matrix} of shape ( (in_data_col-filter_col)/col_stride + 1, (in_data_row-filter_row)/row_stride + 1 )
        """
        cov2D_model = cov2D(input_data=input_data, filter=filter, strides=strides)
        result = cov2D_model.convolution()

        return result

    def __conv(self, input_data, weights, strides, _axis = 1):
        """input tensor conv filter tensor 

        if _axis is 1:
            'input tensor' conv 'filter tensor' to 'result matrix'
        else:
            'input tensor' conv 'filter tensor' to 'result tensor'(do not add up every matrix)
        
        Parameters
        ----------
        input_data : {array-like, tensor(3-dim)} of shape (in_data_col, in_data_row, channel)

        weights : {array-like, tensor(3-dim)} of shape (filter_col, filter_row, filter_channel)

        strides : {tuple-like, vector_2} of shape (col_stride, row_stride)

        _axis : {int-like, scalar} whether to cmpute 2D matrix

        Returns
        -------
        result : if _aixs is 1 : {array-like, matrix} of shape (out_data_col, out_data_row),
                 else : {array-like, tensor(3-dim)} of shape (out_data_col, out_data_row, channel)
        """
        
        output_height = int( (input_data.shape[0] - weights.shape[0]) / strides[0] + 1)
        output_wide = int( (input_data.shape[1] - weights.shape[1]) / strides[1] + 1)


        if _axis == 1:
            
            result = np.zeros((output_height, output_wide))
        
            for channel_index in range(self.__channel):
                result += self.__single_conv(input_data = input_data[:, :, channel_index], filter = weights[:, :, channel_index], strides = strides)

        else:
            result = np.zeros(shape=(output_height, output_wide, self.__channel))
            
            for channel_index in range(self.__channel):
                result[:, :, channel_index] = self.__single_conv(input_data = input_data[:, :, channel_index], filter = weights[:, :, channel_index], strides = strides) 

        return np.array(result)
            
    def forward_pass(self, input_data):
        """forward operation
        
        By multi-layer input tensor conv multi-layer filter tensor, update output tensor

        Parameters
        ----------
        input_data : {array-like, tensor(3-dim)} of shape (in_data_col, in_data_row, channel)

        Returns
        -------
        result : {array-like, tensor(3-dim)} of shape (out_data_col, out_data_row, channel)
        
        """
        self.__input = input_data
        self.__output = self.__conv(input_data=input_data, weights = self.__weights, strides = self.__strides) + self.__b

        return self.__output

    def backward_pass(self, error, learn_rate, activation_name='none'):
        """backward operation

        According to the error matrix,
        1. expand the error matrix to the error tensor 
            of shape(error_col, error_row, in_data_channel)
            by copy the error matrix (every channel has the
            same weight)
        2. expand the X to the tensor with 
            shape(in_data_col + kernel_col - 1, in_data_row + kenel_row - 1, in_data_channel)
            based on the error tensor, input data shape, kernel shape, strides
        3. according to the convolution BP algorithm,
            calculate the single channel situation(No
            accumulation between channels during convolution),
            then get the tensor of shape(in_data_col, in_data_row, in_data_channel)
        4. update weights and bias 

        Parameters
        ----------
        error : {array-like, matrix} of shape (out_data_col, out_data_row)

        learn_rate : {float-like, scalar}

        activation_name : {string-like, scalar} value of {'sigmoid', 'tanh', 'relu', 'none'(default)}

        Returns
        -------
        delta : {array-like, tensor(3-dim)} of shape (in_data_col, in_data_row, in_data_channel)
        """
        
        output_delta = np.zeros( (self.__output_shape[0], self.__output_shape[1], self.__channel))
        
        # step 1: expand the error matrix to the output_delta tensor
        for channel_index in range(self.__channel):
            output_delta[:, :, channel_index] = error
        
        # step 2: init the shape of error tensor after padding
        X = np.zeros(shape=(self.__input_shape[0] + self.__kernel_height - 1, self.__input_shape[1] + self.__kernel_wide - 1, self.__channel))

        # step 2: padding 0    
        for y_index in range(output_delta.shape[0]):
            for x_index in range(output_delta.shape[1]):
                X[ self.__kernel_height - 1 + y_index * self.__strides[0],
                self.__kernel_wide - 1 + x_index * self.__strides[1], :] = output_delta[ y_index, x_index, :]
       
        # step 3: calculate delta of pre layer

        # 'error_{cur_layer}' conv 'rot180(W)' 
        flip_conv_weights = self.__conv( input_data = X, weights = self.__flip_weights(), strides = (1, 1), _axis = 0)
        
        # 'error_{cur_layer-1}' = 'error_{cur_layer}' conv 'rot180(W)' dot-multi 'activation_prime'
        delta = flip_conv_weights * np.reshape( ActivationFunction.activation_prime(activation_name = activation_name, input_data = self.__input), flip_conv_weights.shape)
        
        temp_weights = X[self.__kernel_height-1 : 1-self.__kernel_height, self.__kernel_height-1 : 1-self.__kernel_height, :]
        # step 4 : update weights and bias
        weights_delta = self.__conv(input_data = self.__input, weights = temp_weights , strides=(1, 1), _axis = 0)
    
        self.__update_params(w_delta = weights_delta, b_delta = np.sum(error), learn_rate = learn_rate)
        
        return delta

# input_data_test = np.array(
#     [
#         [1, 3, 5, 4, 7],
#         [2, 3, 2, 1, 0],
#         [7, 8, 1, 2, 3],
#         [3, 2, 9, 8, 7],
#         [2, 3, 4, 0, 2]
#     ]
# )

# cov_filter = np.array(
#     [
#         [0, 0, 0],
#         [0, 1, 0],
#         [0, 0, 0]
#     ]
# )
  
# test_covkernel = CovKernel(kernel_size=(2,2), input_shape=(5,5,2), strides=(1,1))
# # test_covkernel.test_print()
# # test_covkernel.flip_weights()
# print(test_covkernel.conv(input_data=input_data_test, weights=cov_filter, strides=(1,1), padding=1))


        



