import numpy as np

class DataPreprocessorModel(object):
    def __init__(self, X, labels):
        """
        
        训练数据集，将数据分为训练集、验证集、测试集，比例为2：1：1
        并将数据集预处理

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape(n_samples, n_features)
            Training data
        
        labels : {array-like} of shape(n_samples, 1_label)
        
        """
        self.X = X
        self.labels = labels
    
    def __normalize(self):
        # (X-U)/delta
        mean = np.mean(self.X, axis = 0)
        X_norm = self.X - mean
        std = np.std(X_norm, axis = 0)
        X_norm /= std + 10 ** (-5)
        return (X_norm, mean, std)

    def __PCA_white(self, X_train):
        mean  = np.mean(X_train, axis = 0)
        X_norm = X_train - mean
        cov = np.dot(X_norm.T, X_norm)/X_norm.shape[0]
        U, S, V = np.linalg.svd(cov)
        X_norm = np.dot(X_norm, U)
        X_norm /= np.sqrt(S + 10**(-5))
        return (X_norm, mean, U, S)


    
    def __split_data(self):
        
        num_examples = self.X.shape[0]
        shuffle_no = list(range(num_examples))
        np.random.shuffle(shuffle_no)
    
        X_train = self.X[ shuffle_no[:num_examples//2] ]
        labels_train = labels[ shuffle_no[:num_examples//2] ]
    
        X_val = self.X[ shuffle_no[num_examples//2:num_examples//2+num_examples//4]]
        labels_val = labels[ shuffle_no[num_examples//2:num_examples//2+num_examples//4]]
    
        X_test = self.X[shuffle_no[-num_examples//4 :]]
        labels_test = labels[ shuffle_no[-num_examples//4 :]]

        return (X_train, labels_train, X_val, labels_val, X_test, labels_test)



    def __data_preprocess(self, X_train, X_val, X_test):
        (X_train_pca, mean, U, S) = self.__PCA_white(X_train)
        X_val_pca = np.dot(X_val - mean, U)
        X_val_pca /= np.sqrt(S + 10**(-5))
        X_test_pca = np.dot(X_test - mean, U)
        X_test_pca /= np.sqrt(S + 10**(-5))
        return (X_train_pca, X_val_pca, X_test_pca)



    def fit(self):
        """
        
        训练数据集，将数据分为训练集、验证集、测试集，比例为2：1：1
        并将数据集预处理(中心化,规范化,PCA处理)
        
        Returns
        -------
        X_train_pca : 训练集X

        labels_train : 训练集标记集合
        
        X_val_pca : 验证集X

        labels_val : 验证集标记集合

        X_test_pca : 测试集X

        labels_test : 测试集标记集合

        """
        X_train, labels_train, X_val, labels_val, X_test, labels_test = self.__split_data()
        X_train_pca, X_val_pca, X_test_pca = self.__data_preprocess(X_train=X_train, X_val=X_val, X_test=X_test)
        return (X_train_pca, labels_train, X_val_pca, labels_val, X_test_pca, labels_test)
