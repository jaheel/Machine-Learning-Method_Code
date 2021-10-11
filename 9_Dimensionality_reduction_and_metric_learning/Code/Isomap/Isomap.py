import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

def get_k_matrix(data, k):
    """
    近邻矩阵
    @ param data: 样本集
    @ param k: 近邻参数

    @ return k_dist: 近邻矩阵
    """

    dist = pdist(data, 'euclidean') #距离矩阵
    dist = squareform(dist) #转化为方阵
    
    inf = float('inf')
    m = dist.shape[0]
    k_dist = np.ones([m,m])*inf
    for i in range(m):
        top_k = np.argpartition(dist[i], k)[:k+1]
        k_dist[i][top_k] = dist[i][top_k]
    
    return k_dist

def Floyd(data):
    """
    Floyd algorithm 

    @ param data: 距离矩阵(m,m)
    """
    m = data.shape[0]
    for k in range(m):
        for i in range(m):
            for j in range(m):
                data[i][j] = min(data[i][j], data[i][k]+data[k][j])

    return data

def MDS(data, n_dims):
    """

    @ param data: (n_samples, n_features)
    @ param n_dims: target n_dims

    @ return Z: (n_samples, n_dims)

    """
     
    n = data.shape[0]
    dist_ij = data**2

    dist = 1/(n**2)*dist_ij.sum() # (1,1)
    dist_i = np.sum(dist_ij, axis=1, keepdims=True)/n #(n,1)
    dist_j = np.sum(dist_ij, axis=0, keepdims=True)/n #(1,n)

    B = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            B[i][j]= -0.5*(dist_ij[i][j] - dist_i[i][0] - dist_j[0][j] + dist) 

    eig_value, eig_vector = np.linalg.eig(B)
    
    index_list = np.argsort(-eig_value)[:n_dims]
    
    picked_eig_value = eig_value[index_list].real
    picked_eig_vector = eig_vector[:, index_list]

    Z = picked_eig_vector*picked_eig_value**(0.5)
    return Z


def Isomap(data, target_dims, k):
    """
    @ param data : target matrix
    @ param target_dims : target dims
    @ param k : neighbors parameter

    @ return : mds(dist, target_dims)
    """
    k_dist = get_k_matrix(data, k)
    dist = Floyd(k_dist)

    return MDS(dist, target_dims)


if __name__=='__main__':
    
    data =np.array([[1,2,3,4],[2,1,5,6],[3,5,1,7],[4,6,7,1]]) #test data
    outcome = Isomap(data, 2, 3)
    print(outcome)
    
    

