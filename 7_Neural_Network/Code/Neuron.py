import numpy as np
import ActivationFunction as AF
class Neuron(object):
    def __init__(self, weights, bias):
        """
        
        神经元模型

        Parameters
        ----------
        weights : {array-like} of shape(1_sample, n_neuron_layer_before)

        bias : 1_number
        
        """
        self.__weights = weights
        self.__bias = bias
        self.__error = 0
        self.__result = 0

    @property
    def weights(self):
        return self.__weights

    @weights.setter
    def weights(self, value):
        self.__weights = value
    
    @property
    def bias(self):
        return self.__bias

    @bias.setter
    def bias(self, value):
        self.__bias = value

    @property
    def result(self):
        return self.__result
    
    @result.setter
    def result(self, value):
        self.__result = value

    @property
    def error(self):
        return self.__error
    
    @error.setter
    def error(self, value):
        self.__error = value

    

    def update_weights(self, delta):
        """

        更新权重: w_new = w_old + delta

        Parameters
        ----------
        delta : {python list} of shape(n_neurons_layer_before)
        
        """
        self.__weights += delta
    
    def feedforward(self, input_data):
        """
        
        前向运算

        Parameters
        ----------
        input_data : {array-like, sparse matrix} of shape(1_samples, n_neuron_layer_before)
        
        Returns
        -------
        result : 1_number
        
        """
        self.__result = np.dot(self.__weights, input_data) + self.__bias
        return self.__result


    


class Layer(object):
    def __init__(self, neuron_number=(), activate_function = "relu"):
        """
        
        单层神经网络

        Parameters
        ----------
        neuron_number : (1_number_previous_layer_neurons, 1_number_current_layer_neurons)

        activate_function : {1_str}
            1. relu (default)
            2. sigmoid
            3. tanh
            4. leaky_relu
            5. elu
        
        """
        

        self.neuron_number_prev = neuron_number[0]
        self.neuron_number_current = neuron_number[1]

        # 用python list 保存神经元
        self.neurons = [Neuron(weights=self.__weight_init(self.neuron_number_prev), bias=0) for i in range(self.neuron_number_current)]
        
        self.activate_function_str = activate_function    
        self.__AFModel = AF.NNActivator() # 激活函数模型 

        # 双向链表化
        self.next = None
        self.prev = None
    
    
    def feedforward(self, input_data):
        """

        feed forward

        Parameter
        ---------
        input_data : {array-like, np.array} of shape(1_samples, n_activate_result)

        Returns
        -------
        result : {array-like, python_list } of shape(1_samples, n_activate_result)
        
        """
        input_data = np.array(input_data)
        self.input_data = input_data

        result = []

        for element in self.neurons:
            result.append( element.feedforward(self.input_data))

        result = self.__AFModel.fit(data=result, function_name=self.activate_function_str)
        
        #将激活后的结果保存到神经元中
        for index in range(len(result)):
            self.neurons[index].result = result[index]

        return result
    
    def __weight_init(self, in_neuron_numbers):
        """

        小随机数初始化

        Parameters
        ----------
        in_neuron_numbers : (1_number) the number of neurons in the previous layer

        Returns
        -------
        weights : {array-like, np.array } of shape(1_samples, n_in_depth)

        """
        weights = np.random.randn(in_neuron_numbers) / np.sqrt(in_neuron_numbers)
        
        return weights

    def get_weights(self):
        for i in range(self.neuron_number_current):
            print(self.neurons[i].weights)

    
    def error_update(self, error_update_list):
        """

        更新神经元的误差

        Parameters
        ----------
        error_update_list : {array-like, python_list} of shape(1_samples, n_neuron_error)

        """

        for index_cur in range(self.neuron_number_current):
            self.neurons[index_cur].error = error_update_list[index_cur]

    
        
class Network(object):
    def __init__(self, input_data, labels, learn_rate = 0.01, mse_error_rate = 0.01, max_iter = 100000):
        """

        Parameters
        ----------
        input_data : {array-like, python_list } of shape(1_samples, n_in_depth)

        labels : {array-like, python_list} of shape(1_samples, n_results)

        learn_rate : (1_number) BP algorithm learn rate

        mse_error : (1_number) range(0,1) (default: 0.01)

        max_iter : the max number of iterations(default: 100000)

        """
        self.input_data = input_data
        self.labels = labels
        self._learn_rate = learn_rate
        self.__mse_error_rate = mse_error_rate
        self.__max_iter = max_iter

        self._head = None
        self._tail = None
        
    def is_empty(self):
        """

        判断网络链表是否为空

        """
        return self._head is None

    def length(self):
        """

        网络层数

        Returns
        -------
        count : (1_number) the number of network layers

        """
        cur = self._head
        count = 0

        while cur is not None:
            count += 1
            cur = cur.next
        
        return count
  
    def items(self):
        """

        遍历网络链表

        """
        cur = self._head

        while cur is not None:
            # 返回生成器
            yield cur
            # 指针下移
            cur = cur.next
   
    def add_layer(self, layer):
        """

        添加Layer

        Parameters
        ----------
        layer : simple net layer
        
        """
        node = layer
        
        if self.is_empty():
            self._head = node
            self._tail = node
        else:    
            # 新结点上一级指针指向旧尾部
            node.prev = self._tail
            self._tail.next = node
            self._tail = self._tail.next


    def __feedforward(self):
        """

        前馈神经网络, 前向运算
        
        """
        out_result = self.input_data

        if self.is_empty():
            pass

        else:
            cur = self._head
            while cur is not None:
                out_result = cur.feedforward(out_result)
                cur = cur.next
        
        self.result = out_result

    def __backward(self):
        """

        反向传播, 更新误差、权重

        """
        self.__error_update()
        self.__weights_update()

    def predict(self):
        """

        预测

        """
        iter_number = 0
        
        self.__feedforward()
        

        while self.__cost_function_MSE() > self.__mse_error_rate and iter_number < self.__max_iter:
            
            iter_number += 1
            self.__backward()
            self.__feedforward()

            self.print_weights()
            print("----result-----")
            print(self.result)
        

    def print_weights(self):
        cur = self._head
        while cur is not None:
            print("----weights-----")
            cur.get_weights()
            cur = cur.next

    def __cost_function_MSE(self):
        """

        计算MSE损失函数 : total_error = 0.5 * (origin_y - predict_y)

        Returns
        -------
        total_error : (1_number)

        """
        mean = self.labels - self.result
        total_error = 0

        for element in mean:
            total_error += 0.5 * element * element
        
        print("----total error-----")
        print(total_error)

        return total_error

    def __error_update(self):
        """
        
        每一层神经网络的误差更新

        """
        cur = self._tail
        error_list = self.labels - self.result

        

        # 从后往前，层层递进更新error
        while cur is not None:
            cur.error_update(error_list)
            error_update_list = []
            
            
            for index_pre in range(cur.neuron_number_prev):
                temp_result = 0
                
                for index_cur in range(cur.neuron_number_current):
                   
                    temp_result += cur.neurons[index_cur].weights[index_pre] * error_list[index_cur]
                
                error_update_list.append(temp_result)

            cur = cur.prev
            error_list = error_update_list

    
    def __weights_update(self):
        """

        每一层网络，每一个神经元的权值更新

        """
        cur = self._tail
        pre = self._tail.prev
        
        while cur is not None:

            # pre is None 代表cur这一层是第一层神经元
            if pre is None:
                for index_cur in range(cur.neuron_number_current):
                    
                    weights_update_list = []
                    for index_pre in range(len(self.input_data)):
                        
                        # 每一个w的更新: w_new = w_old + mu * error * x_n
                        w_update = self._learn_rate * cur.neurons[index_cur].error * self.input_data[index_pre]
                        
                        weights_update_list.append(w_update)
                    
                    cur.neurons[index_cur].update_weights(weights_update_list)
                
                cur = cur.prev
            
            else:
                for index_cur in range(cur.neuron_number_current):
        
                    weights_update_list = []
                    
                    for index_pre in range(cur.neuron_number_prev):
        
                        # 每一个w的更新: w_new = w_old + mu * error * x_n
                        w_update = self._learn_rate * cur.neurons[index_cur].error * pre.neurons[index_pre].result
                        
                        weights_update_list.append(w_update)
                    
                    cur.neurons[index_cur].update_weights(weights_update_list)
                
                cur = cur.prev
                if cur.prev is None:
                    pre = None
                else:
                    pre = cur.prev
            
            
            

            
            

                 
                    
                
        

        



        
    
    

        


input_data = np.array([-1,1,2,4], [3, 2, 3, 5])
label = np.array([1,0])

test_network = Network(input_data=input_data, labels=label)
test_network.add_layer(Layer(neuron_number=(4,2), activate_function="sigmoid"))
test_network.add_layer(Layer(neuron_number=(2,3), activate_function="sigmoid"))
test_network.add_layer(Layer(neuron_number=(3,2), activate_function="sigmoid"))
test_network.predict()
