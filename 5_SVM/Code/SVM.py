# -*- coding: utf-8 -*-
"""

Author: Fons Hui
Edit: 2020-8-26
Email: xufanxin86@gmail.com

Introduction: Simple implementation of a SVM using the SMO algorithm


"""

import numpy as np
import random

class SVM():
    
    def __init__(self, max_iteration=10000, kernel_type='linear', C=1.0, epsilon=0.001):
        self.max_iteration=max_iteration
        self.kernels={
            'linear': self.kernel_linear,
            'quadratic': self.kernel_quadratic
        }
        self.kernel_type=kernel_type
        self.C=C
        self.epsilon=epsilon
    
    #  function of kernel type 
    def kernel_linear(self, x_1, x_2):
        return np.dot(x_1,x_2.T)
    
    def kernel_quadratic(self, x_1, x_2):
        return(np.dot(x_1,x_2.T)** 2)
    
    # random select
    def get_random_int(self, begin, end, i):
        select_j=i
        while select_j==i:
            select_j=int(random.uniform(begin,end))
        return select_j

    def compute_L_H(self, C, alpha_old_j, alpha_old_i, y_j, y_i):
        if(y_i!=y_j):
            return(max(0, alpha_old_j - alpha_old_i),min(C, C - alpha_old_i + alpha_old_j))
        else:
            return(max(0, alpha_old_i + alpha_old_j - C),min(C, alpha_old_i + alpha_old_j))

    def calculate_w(self, alpha, y, X):
        return np.dot(X.T,np.multiply(alpha,y))

    def calculate_b(self, X, y, w):
        b_temp=y-np.dot(w.T,X.T)
        return np.mean(b_temp)

    def h(self, X, w, b):
        return np.sign(np.dot(w.T,X.T) + b).astype(int)
    
    def predict(self, X):
        return self.h(X, self.w, self.b)

    def E(self, x_k, y_k, w, b):
        return self.h(x_k, w, b) - y_k

    def clip_alpha(self,alpha_i,L,H):
        alpha_i = max(alpha_i, L)
        alpha_i = min(alpha_i, H)
        return alpha_i
    
    def fit(self,X,y):
        #column of n
        n=X.shape[0]
        alpha=np.zeros((n))
        
        #select kernel type
        kernel=self.kernels[self.kernel_type]

        iteration_count=0

        while True:
            iteration_count+=1
            
            alpha_old=np.copy(alpha)

            #SMO algorithm, 
            for j in range(0,n):
                #random select i
                i=self.get_random_int(0,n-1,j)
                
                x_i=X[i,:]
                x_j=X[j,:]
                y_i=y[i]
                y_j=y[j]

                mu=kernel(x_i, x_i) + kernel(x_j, x_j) - 2*kernel(x_i, x_j)
                if mu == 0:
                    continue

                alpha_old_j=alpha[j]
                alpha_old_i=alpha[i]

                (L,H)=self.compute_L_H(self.C, alpha_old_j, alpha_old_i, y_i, y_j)
                
                self.w=self.calculate_w(alpha, y, X)
                self.b=self.calculate_b(X, y, self.w)

                E_i=self.E(x_i, y_i, self.w, self.b)
                E_j=self.E(x_j, y_j, self.w, self.b)

                # Update alpha_new_j
                alpha[j]=alpha_old_j + float(y_j * (E_i - E_j))/mu
                alpha[j]= self.clip_alpha(alpha[j],L,H)
                
                # Update alpha_new_i
                alpha[i]=alpha_old_i + y_i*y_j * (alpha_old_j - alpha[j])
            
            # L2
            difference=np.linalg.norm(alpha-alpha_old)

            if(difference<self.epsilon):
                break

            # Iteration number exceeded the max number of iterations
            if(iteration_count>= self.max_iteration):
                return
        
        self.b=self.calculate_b(X,y,self.w)

        if self.kernel_type== 'linear' :
            self.w = self.calculate_w(alpha, y, X)
        
        # get the index of support vectors
        alpha_index=np.where(alpha > 0)[0]
        support_vectors = X[alpha_index, :]

        return support_vectors
            

                




                


