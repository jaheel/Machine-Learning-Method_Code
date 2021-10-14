import numpy as np
from Layer import Layer
from padding import Padding2D
from ConvKernel import ConvKernel
from activation import ActivationFunction

class ConvLayer(Layer):
    def __init__(self, filters, kernel_size, input_shape, strides, padding="SAME", activation="none", name="conv"):
        """

        Parameters
        ----------
        filters : {int-like, scalar} the number of filters
        
        kernel_size : {tuple-like, vector_2} of shape (filter_col, filter_row), the size of single filter

        input_shape : {tuple-like, vector_3} of shape (in_data_col, in_data_row, channel) ep: (64,64,3)

        strides : {tuple-like, vector_2} of shape (col_stride, row_stride) (ep: (1,1))

        padding : {string-like, scalar} padding patern, value of { 'SAME'(default), 'VALID'}

        activation : {string-like, scalar} value of {'sigmoid', 'tanh', 'relu', 'none'(default)}

        name : {string-like, scalar} current layer name
         
        """
        super().__init__()
        self.__filters = filters
        self.__kernel_size = kernel_size
        self.__strides = strides
        self.__padding = padding
        self.activation_name = activation
        self.__input_shape = input_shape
        self.__input_padding_shape = input_shape
        self.__input = np.zeros(self.__input_shape)
        self.name = name
        self.flag = False # kernels whether init

    
    def __padding_in_data(self, in_data):
        """
        
        Parameters
        ----------
        in_data : {array-like, tensor(3-dim)} of shape (in_data_col, in_data_row, channel)

        Returns
        -------
        result : {array-like, tensor(3-dim)} of shape (in_data_col + padding_col, in_data_row + padding_row, channel) after padding
        """

        result = Padding2D.padding_tensor(input_data = in_data, filter_size = self.__kernel_size, strides = self.__strides, padding = self.__padding)
        
        return result

    def __output_init(self, in_data_shape):
        """init output tensor

        Parameters
        ----------
        in_data_shape : {tuple-like, vector_3} of shape (in_data_col, in_data_row, in_data_channel)
        """
        output_height = int( (in_data_shape[0] - self.__kernel_size[0] ) / self.__strides[0] + 1 )
        output_wide = int( (in_data_shape[1] - self.__kernel_size[1] ) / self.__strides[1] + 1 )
        
        self.__output_size = (output_height, output_wide,  self.__filters)
        self.__output = np.zeros(self.__output_size)
    
    def forward_propagation(self, _in_data):
        """forward propagation

        Parameters
        ----------
        _in_data : {array-like, tensor(3-dim)} of shape(in_data_col, in_data_row, in_data_channel)
        """
        self.__input = self.__padding_in_data(in_data = _in_data)
        self.__input_padding_shape = self.__input.shape

        self.__output_init(in_data_shape = self.__input_padding_shape)
        

        if not self.flag:
            self.__kernels = [ConvKernel(kernel_size = self.__kernel_size, input_shape = self.__input_padding_shape, strides = self.__strides) for _ in range(self.__filters) ]
            self.flag = True

        for kernel_index, kernel in enumerate(self.__kernels):
            
            self.__output[:, :, kernel_index] = kernel.forward_pass(input_data = self.__input)
        

        
        return ActivationFunction.activation(input_data = self.__output, activation_name = self.activation_name)

    def back_propagation(self, error, learn_rate):
        """backward propagation

        Before compute previous layer error average,
        We should restore the data size to the input
        shape.

        Parameters
        ----------
        error : {array-like, tensor(3-dim)} of shape (out_data_col, out_data_row, kernel_channel)

        learn_rate : {float-like, scalar}

        Returns
        -------
        error_pre : {array-like, tensor(3-dim)} of shape (in_data_col, in_data_row, in_data_channel)
                    previous layer error (the average of the tensor by the kerner tensor calculated)

        """
        
        delta = np.zeros(self.__input_shape)

        for kernel_index in range(self.__filters):
            index = self.__filters - kernel_index - 1

            temp = self.__kernels[index].backward_pass(error = error[:, :, index], learn_rate = learn_rate, activation_name = self.activation_name)

            if self.__padding == 'VALID':
                bd = np.ones(self.__input_shape)
                bd[ :self.__input_padding_shape[0], :self.__input_padding_shape[1] ] = temp
            
            elif self.__padding == 'SAME':
                pad_add_number_h = int((self.__kernel_size[0] - 1) / 2 )
                pad_add_number_w = int((self.__kernel_size[1] - 1) / 2 )
                bd = temp[pad_add_number_h : pad_add_number_h + self.__input_shape[0], pad_add_number_w : pad_add_number_w + self.__input_shape[1]]
            
            else:
                raise ValueError("padding name is wrong")

            delta += bd
        
        return delta / self.__filters