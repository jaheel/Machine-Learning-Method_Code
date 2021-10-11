"""
@ author: Alex
@ E-mail: xufanxin86@gmail.com

@ Introduction: Simple application of PCA algorithm 

"""

import numpy as np



def PCA(X,k):
    """
    @ param X : data set
    @ param k : the components you want
    @ return data : k dimension of features
    """
    # step 1: mean of each feature
    samples_number, features_number = X.shape
    mean = np.array([np.mean(X[:,i]) for i in range(features_number)])

    normalization_X=X-mean
    
    # step 2: find the scatter matrix
    # scatter matrix is same to the convariance matrix, scatter_matrix =  convariance_matrix * (data - 1)
    scatter_matrix = np.dot(np.transpose(normalization_X), normalization_X)

    # step 3: Calculate the eigenvectors and eigenvalues
    eig_values, eig_vectors = np.linalg.eig(scatter_matrix)

    # step 4: sort eig_vectors based on eig_values from highest to lowest
    eig_pairs = [(np.abs(eig_values[i]), eig_vectors[:,i]) for i in range(features_number)]
    eig_pairs.sort(reverse = True,key=(lambda x: x[0]))

    # step 5: select the top k eig_vectors
    feature = np.array([element[1] for element in eig_pairs[:k]])

    # step 6: get new data
    data = np.dot( normalization_X, np.transpose(feature))

    return data

# test  
X = np.array([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
data=PCA(X,1)
print(data)