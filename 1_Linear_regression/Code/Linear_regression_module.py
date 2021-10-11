import numpy as np

class LinearRegression():
    def __init__(self):
        self.w = None

    
    def __featureNormaliza(self, X):
        """
        
        Normalized feature dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape(n_samples, n_features)
            Training data
        
        Returns
        -------
        X_norm : norm data
        
        mu : mean of every row

        sigma : standard deviation of every row
        """

        X_norm = np.array(X)

        mu = np.mean(X_norm, 0) # average
        sigma = np.std(X_norm, 0) # standard deviation

        for i in range(X.shape[1]):
            X_norm[:, i] = (X_norm[:,i] - mu[i])/sigma[i]
        
        return X_norm

    def __gradientDescent(self, X, y, alpha=0.001, max_iteration_count=10000):
        """

        batch gradient descent
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape(n_samples, n_features)
            Training data
        
        y : array-like of shape(n_samples,) or (n_samples, n_targets)
            Target values.
        
        alpha : learning rate

        Returns
        -------
        w : the final value list formed after gradient descent

        J_history : w change log
        """
        y = np.array(y)
        m,n = X.shape
        w = np.ones((n))

        iteration_count = 0
     
        J_history = []
        w_change_log = []


        while iteration_count < max_iteration_count:

            X_w_ip = np.dot(X, w) # w^T * X
            
            temp_d = X_w_ip.flatten() - y # h_theta(X) - y
            temp_dot = np.dot(np.transpose(X), temp_d) # (h_theta(X) - y)*X
            temp = w - ((alpha/m) * temp_dot)
            w = temp
            
            J = np.dot((np.dot(X,w) - y) , (np.transpose(np.dot(X,w) - y)))/(2 * m) # cost J
            J_history.append(J)
            iteration_count += 1
        
        self.w = w
        return w, J_history  
            

    def fit(self, X, y):
        """

        Fit linear model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape(n_samples, n_features)
            Training data

        y : array-like of shape(n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """
        X = self.__featureNormaliza(X)
        X = np.insert(X, 0, 1, axis=1)

        # X_inv = np.linalg.inv(X.T.dot(X)) # X^T.dot(X)
        # self.w = X_inv.dot(X.T).dot(y)
        
        w, J_history = self.__gradientDescent(X, y)
        # print(J_history)


    



    def predict(self, X):
        """

        Predict using the linear model.

        Parameters
        ----------
        X : array-like or sparse matrix, shape(n_samples, n_features)
            Samples.
        
        Returns
        -------
        C : array, shape [n_samples]
            Predicted class label per sample.
        """
        # h(theta) = theta.T.dot(X)
        X = self.__featureNormaliza(X)
        X = np.insert(X, 0, 1, axis=1)
        C = X.dot(self.w)
        return C



