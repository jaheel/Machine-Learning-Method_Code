import numpy as np
import matplotlib.pyplot as plt
import sys
 
def distancematrix(test):
    leng=len(test)
    resmat=np.zeros([leng,leng],np.float32)
    for i in range(leng):
      for j in range(leng):
          resmat[i,j]=np.linalg.norm(test[i]-test[j])
    return resmat #返回距离矩阵
def mds(test,deg):
    length= len(test)
    re= np.zeros((length, length),np.float32)
    if(deg>length):
        deg=length
    D= distancematrix(test)
    ss = 1.0 /length ** 2 * np.sum(D ** 2)
    for i in range(length):
        for j in range(length):
            re[i, j] = -0.5 * (D[i, j] ** 2 - 1.0 / length * np.dot(D[i, :], D[i, :]) - 1.0 / length * np.dot(D[:, j], D[:, j]) + ss)
 
    A, V = np.linalg.eig(re)
    list_idx = np.argpartition(A, deg- 1)[-deg:]
    a = np.diag(np.maximum(A[list_idx], 0.0))
    return np.matmul(V[:, list_idx], np.sqrt(a))
# 使用 Dijkstra 算法获取最短路径，并更新距离矩阵
# test: 距离矩阵，大小 m * m
# start：最短路径的起始点，范围 0 到 m-1
def usedijk(test, start):
    count = len(test)
    col= test[start].copy()
    rem = count - 1
    while rem > 0:
        i= np.argpartition(col, 1)[1]
        length = test[start][i]
        for j in range(count):
            if test[start][j] > length + test[i][j]:
                test[start][j] = length + test[i][j]
                test[j][start] = test[start][j]
        rem -= 1
        col[i] = float('inf')
 
# isomap 算法的具体实现
# test：需要降维的矩阵
# target：目标维度
# k：k 近邻算法中的超参数
# return：降维后的矩阵
def isomap(test, target, k):
    inf = float('inf')
    count = len(test)
    if k >= count:
        raise ValueError('K is too large')
    mat_distance = distancematrix(test)
    knear = np.ones([count, count], np.float32) * inf
    for idx in range(count):
        topk = np.argpartition(mat_distance[idx], k)[:k + 1]
        knear[idx][topk] = mat_distance[idx][topk]
    for idx in range(count):
        usedijk(knear, idx)
    return mds(knear, target)
 
if __name__ == '__main__':
    print('开始降维.....')
    
    D =np.array([[1,2,3,4],[2,1,5,6],[3,5,1,7],[4,6,7,1]]) #test data
    outcome= isomap(D, 2, 3)
    sys.stdout.write('降维完成\n')
    print(outcome)