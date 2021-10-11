
import numpy as np
import matplotlib.pyplot as plt

# generate data

class DataGeneratorModel(object):
    
    def __init__(self, dim=2, N_class=4, num_sample_per_class = 200):
        """
        
        数据生成器模型

        Parameters
        ----------
        dim : feature number (default : 2)

        N_class : label number (default : 4)

        num_sample_per_class : the number of samples in every class (default : 200)
        
        Returns
        -------
        

        """
        self.dim = dim
        self.N_class = N_class
        self.num_sample_per_class = num_sample_per_class
    
    def __gen_toy_data(self):
    
        num_examples = self.num_sample_per_class * self.N_class # sample number
        X = np.zeros((num_examples, self.dim))
        labels = np.zeros(num_examples, dtype= 'uint8')
    
        for j in range(self.N_class):
            ix = range(self.num_sample_per_class * j, self.num_sample_per_class * (j+1))
            x = np.linspace(-np.pi, np.pi, self.num_sample_per_class) + 5
            y = np.sin(x + j * np.pi/(0.5 * self.N_class))
            y += 0.2 * np.sin(10*x + j*np.pi/(0.5*self.N_class))
            y += 0.25*x + 10 
            y += np.random.randn(self.num_sample_per_class) * 0.1 #noise
        
            X[ix] = np.c_[x, y]
            labels[ix] = j
    
        return (X, labels)


        
    def show_data(self, X, labels):
        plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap=plt.cm.Spectral)
        plt.show()

    
    def fit(self):
        return self.__gen_toy_data()


    






