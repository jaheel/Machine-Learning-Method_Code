import numpy as np



#-----------------------
# test padding2D:
#
# from padding import Padding2D
#
# input_data_test = np.array(
#     [
#         [1, 3, 5, 4, 7],
#         [2, 3, 2, 1, 0],
#         [7, 8, 1, 2, 3],
#         [3, 2, 9, 8, 7],
#         [2, 3, 4, 0, 2]
#     ]
# )

# print(Padding2D.padding_data2D(input_data = input_data_test, filter_size = 1))
#--------------------------

# ---------------------------
# test ConvLayer
#
# from ConvLayer import ConvLayer

# test_data = np.random.randint(2, size = (5,5,2))
# # print(test_data)

# test_layer = ConvLayer(filters = 1, kernel_size = (3, 3), input_shape=(5, 5, 2), strides=(1, 1), padding = 'SAME', activation = "relu")
# result = test_layer.forward_propagation(_in_data = test_data)
# print("result : ", result)
# print("result shape: ", result.shape)
# error_data = np.random.randint(2, size = result.shape)
# print("error : ", error_data)
# print(test_layer.back_propagation(error=error_data, learn_rate = 0.01))

# ----------------------------