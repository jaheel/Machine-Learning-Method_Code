from ConvLayer import ConvLayer
from DenseLayer import DenseLayer
from FlattenLayer import FlattenLayer
from PoolLayer import PoolLayer

def Conv(filters, kernel_size, input_shape, strides = (1, 1), padding = "SAME", activation = "none", name = "conv"):
    """ConvLayer

    Parameters
    ----------
    filters : {int-like, scalar} the number of filters
        
    kernel_size : {tuple-like, vector_2} of shape (filter_col, filter_row), the size of single filter

    input_shape : {tuple-like, vector_3} of shape (in_data_col, in_data_row, channel) ep: (64,64,3)

    strides : {tuple-like, vector_2} of shape (col_stride, row_stride) (ep: (1,1))

    padding : {string-like, scalar} padding patern, value of { 'SAME'(default), 'VALID'}

    activation : {string-like, scalar} value of {'sigmoid', 'tanh', 'relu', 'none'(default)}

    name : {string-like, scalar} current layer name, default is 'conv'
    
    Returns
    -------
    result : {ConvLayer-like, scalar}
    """
    return ConvLayer(filters = filters, kernel_size = kernel_size, input_shape = input_shape, strides = strides, padding = padding, activation = activation, name = name)


def Dense(shape, activation = 'relu', name = 'dense'):
    """
    Parameters
    ----------
    shape : {tuple-like, vector_2} of shape (in_neurals, out_neurals)

    activation : {string-like, scalar} value of {'sigmoid', 'tanh', 'relu'(default), 'softmax','none'}

    name : {string-like, scalar} current layer name

    Returns
    -------
    result : {DenseLayer-like, scalar}
    """

    return DenseLayer(shape = shape, activation = activation, name = name)


def Pool(input_shape, filter_size = (2, 2), strides = (2, 2), pool_method = 'MAX', name = 'none'):
    """PoolLayer

    Parameters
    ----------
    input_shape : {tuple-like, vector_3} of shape (in_data_col, in_data_row, in_data_channel)

    filter_size : {tuple-like, vector_2} of shape (filter_col, filter_row) default is (2, 2)

    strides : {tuple-like, vector_2} of shape (col_stride, row_stride) default is (2, 2)

    pool_method : {string-like, scalar} value of {'MAX'(default), 'AVG'}

    name : {string-like, scalar} default is 'none'
    
    Returns
    -------
    result : {PoolLayer-like, scalar}
    """
    
    return PoolLayer(input_shape = input_shape, filter_size = filter_size, strides = strides, pool_method = pool_method, name = name)


def Flatten():
    return FlattenLayer()