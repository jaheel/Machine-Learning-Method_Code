# -*- coding: utf-8 -*-
"""
Author: Fons Hui
Edit: 2020-8-26
Email: xufanxin86@gmail.com

Introduction: test SVM

"""

import os
import csv
import sys
import numpy as np
from SVM import SVM

def read_data(filename,header=True):
    data=[]
    header=None
    
    with open(filename, 'r') as csvfile:
        spamreader=csv.reader(csvfile, delimiter=',') # delimiter: '\t' ',' (refer to the dataSet)
        
        if header:
            header= spamreader.next()
        
        for row in spamreader:
            data.append(row)
    
    return (np.array(data), np.array(header))


def calculate_accuracy(y_prime, y_test):
    index=np.where(y_test == 1)
    TP=np.sum(y_test[index]==y_prime[index])

    index=np.where(y_test == -1)
    TN=np.sum(y_test[index] == y_prime[index])

    return float(TP + TN)/len(y_prime)

def main():
    # iris-slwc.txt
    # iris-versicolor.txt
    # iris-virginica.txt
    # testSet.txt
    # Load Data
    filename='data/iris-slwc.txt'
    filepath=os.path.dirname(os.path.abspath(__file__))
    (data,_)=read_data('%s/%s' %(filepath,filename),header=False)
    data=data.astype(float)
    
    # Split data
    X=data[:,0:-1]
    y=data[:,-1].astype(int)

    # Fit model
    model=SVM()
    support_vectors=model.fit(X,y)
    
    support_vectors_count=support_vectors.shape[0]

    y_test=model.predict(X)
    accuracy=calculate_accuracy(y,y_test)

    print("Support vector count: %d" % (support_vectors_count))
    print("bias:\t\t%.3f" % (model.b))
    print("w:\t\t" + str(model.w))
    print("accuracy:\t%.3f" % (accuracy))

if __name__ == '__main__':
    main()