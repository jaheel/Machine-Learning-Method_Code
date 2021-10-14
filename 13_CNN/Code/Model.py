import numpy as np

class Model(object):
    def __init__(self):
        """Model init

        Use python list to store every layer
        
        """
        self.layers = []


    def add(self, layer):
        """add layer

        Parameters
        ----------
        layer : {Layer-like, scalar}
        """

        self.layers.append(layer)


    def __mse(self, predict_y, y, is_forward):
        """mean square error of single sample

        Parameters
        ----------
        predict_y : {array-like, vector} of shape (sample_label_number)

        y : {array-like, vector} of shape (sample_label_number)

        is_forward : {bool-like, scalar} whether is forward propagation
        
        Returns
        -------
        loss_result : {array-like, vector} of shape (sample_label_number)
        """
        
        if is_forward:
            return 0.5 * ((predict_y - y) ** 2)
        
        else: # the delta
            return predict_y - y

    
    def __cross_entropy(self, predict_y, y, is_forward):
        """cross entropy error of single sample

        Parameters
        ----------
        predict_y : {array-like, vector} of shape (sample_label_number)

        y : {array-like, vector} of shape (sample_label_number)

        is_forward : {bool-like, scalar} whether is forward propagation
        
        Returns
        -------
        loss_result : {array-like, vector} of shape (sample_label_number)
        """

        predict_y[predict_y == 0] =1e-12
        
        if is_forward:
            return - y * np.log(predict_y)
        
        else: # backward delta
            return - y / predict_y
    
    def __final_loss(self, loss):
        """compute final loss
        
        Parameters
        ----------
        loss : {array-like, vector} of shape (sample_label_number)

        Returns
        -------
        result : {float-like, scalar}
        """
        
        return np.squeeze(np.mean(loss))

    def set_loss_function(self, loss_function):
        """set final loss function
        
        Parameters
        ----------
        loss_function : {string-like, scalar} value of {'mse', 'cross_entropy'}

        """
        if loss_function == 'mse':
            self.__loss_function = self.__mse
        
        elif loss_function == 'cross_entropy':
            self.__loss_function = self.__cross_entropy
        
        else:
            ValueError("loss function name is wrong")


    def train(self, X, y, learn_rate, epochs):
        """train network

        Parameters
        ----------
        X : {array-like, tensor(4-dim)} of shape (sample_number, in_data_col, in_data_row, in_data_channel)

        y : {array-like, matrix} of shape (sample_number, sample_label_number)

        learn_rate : {float-like, scalar} 

        epochs : {int-like, scalar} dataset learning times
        """
        
        if self.__loss_function is None:
            raise Exception("set loss function first")

        
        for epoch_index in range(epochs):
            
            loss = 0

            for sample_index in range(len(X)):
                single_train_sample_out = X[sample_index]

                # forward 
                for layer in self.layers:
                    single_train_sample_out = layer.forward_propagation(single_train_sample_out)
                
                loss += self.__loss_function(predict_y = single_train_sample_out, y = y[sample_index], is_forward = True)
                error = self.__loss_function(predict_y = single_train_sample_out, y = y[sample_index], is_forward = False)

                # backward
                for j in range(len(self.layers)):
                    layer_index = len(self.layers) - j - 1
                    error = self.layers[layer_index].back_propagation(error, learn_rate)
                
            print("epochs {} / {}  loss : {}".format(epoch_index, epochs, self.__final_loss(loss / len(X))))


    def train_eval(self, X, y, learn_rate, epochs, X_test, y_test):
        """train network with acc of test

        
        Parameters
        ----------
        X : {array-like, tensor(4-dim)} of shape (sample_number, in_data_col, in_data_row, in_data_channel)

        y : {array-like, matrix} of shape (sample_number, sample_label_number)

        learn_rate : {float-like, scalar} 

        epochs : {int-like, scalar} dataset learning times

        X_test : {array-like, tensor(4-dim)} of shape (test_sample_number, in_data_col, in_data_row, in_data_channel)

        y_test : {array-like, matrix} of shape (test_sample_number, sample_label_number)

        """

        if self.__loss_function is None:
            raise Exception("set loss function first")

        for epoch_index in range(epochs):
            loss = 0
            print(loss)
            for sample_index in range(len(X)):
                single_sample_train_out = X[sample_index]

                # forward
                for layer in self.layers:
                    single_sample_train_out = layer.forward_propagation(single_sample_train_out)

                loss += self.__loss_function(predict_y = single_sample_train_out, y = y[sample_index], is_forward = True)
                error = self.__loss_function(predict_y = single_sample_train_out, y = y[sample_index], is_forward = False)

                # backward
                for j in range(len(self.layers)):
                    layer_index = len(self.layers) - j - 1
                    error = self.layers[layer_index].back_propagation(error, learn_rate)
            
            result = self.predict(X_test)
            acc_sample_number = 0

            for sample_index, y_single_sample in enumerate(y_test):
                predict_label_index = np.argmax(result[sample_index])
                single_sample_label_index = np.argmax(y_single_sample)

                if predict_label_index == single_sample_label_index:
                    acc_sample_number += 1
            
            print("epochs {} / {}  loss : {}  acc : {}".format(epoch_index, epochs, self.__final_loss(loss / len(X)), acc_sample_number / len(y_test)))
        
    def predict(self, X):
        """predict result

        Parameters
        ----------
        X : {array-like, tensor(4-dim)} of shape (sample_number, in_data_col, in_data_row, in_data_channel)

        Returns
        -------
        result : {array-like, matrix} of shape (sample_number, sample_label_predict_number)
        """
        
        result = []

        for sample_index in range(len(X)):
            single_sample_train_out = X[sample_index]

            for layer in self.layers:
                single_sample_train_out = layer.forward_propagation(single_sample_train_out)
            
            result.append(single_sample_train_out)
        
        return np.squeeze(np.array(result))