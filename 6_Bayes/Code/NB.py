# _*_ coding:utf-8 _*_
import numpy as np

def load_data_set():
    """
    导入数据， 1代表脏话
    @ return posting_list: 数据集
    @ return class_vectors: 分类向量
    """
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vectors = [0, 1, 0, 1, 0, 1]  
    return posting_list, class_vectors

# 汇总所有词汇，词汇只需要统计一次性
def create_vocabulary_list(data_set):
    """
    创建词库
    @ param data_set: 数据集
    @ return vocabulary_set: 词库
    """
    vocabulary_set = set([])
    for document in data_set:
        # 求并集
        vocabulary_set = vocabulary_set | set(document)
    return list(vocabulary_set)

def words_2_vector(vocabulary_list, input_set):
    """
    文本词向量.词库中每个词当作一个特征，文本中就该词，该词特征就是1，没有就是0
    @ param vocabulary_list: 词表
    @ param input_set: 输入的数据集
    @ return return_vector: 返回的向量，统计input_set中各个词汇是否在vocabulary_set中
    """
    return_vector = [0] * len(vocabulary_list)
    for word in input_set:
        if word in vocabulary_list:
            return_vector[vocabulary_list.index( word)] = 1
        else:
            print("单词: %s 不在词库中!" % word)
    return return_vector


def train_NB(train_matrix, train_category):
    """
    训练
    @ param train_matrix: 训练集
    @ param train_category: 分类
    """
    train_document_number = len(train_matrix)
    words_number = len(train_matrix[0])
    probability_abusive = sum(train_category) / float(train_document_number)
    
    #防止某个类别计算出的概率为0，导致最后相乘都为0，所以初始词都赋值1，分母赋值为2.(拉普拉斯修正)
    p0_number = np.ones(words_number) 
    p1_number = np.ones(words_number)
    p0_denominator = 2
    p1_denominator = 2
    
    for i in range(train_document_number):
        if train_category[i] == 1:
            p1_number += train_matrix[i]
            p1_denominator += sum(train_matrix[i])
        else:
            p0_number += train_matrix[i]
            p0_denominator += sum(train_matrix[i])
    
    # 这里使用log函数，方便计算，因为最后是比较大小，所有对结果没有影响。
    p_theta1 = np.log(p1_number / p1_denominator)
    p_theta0 = np.log(p0_number / p0_denominator)
    
    return p_theta0, p_theta1, probability_abusive

def classify_NB(test_word_vector, p_theta0, p_theta1, probability_abusive):
    """
    判断大小,贝叶斯判别，log运算，将乘法变成加法
    """
    p1 = sum(test_word_vector * p_theta1)+np.log(probability_abusive)
    p0 = sum(test_word_vector * p_theta0)+np.log(1-probability_abusive)
    if p1 > p0:
        return 1
    else:
        return 0

def testing_NB():
    
    posting_list, class_vectors = load_data_set()
    my_vocabulary_list = create_vocabulary_list(posting_list)
    
    # 各个数据集在词汇表中是否出现的向量统计
    train_mat = []
    for posting_document in posting_list:
        train_mat.append(words_2_vector(my_vocabulary_list, posting_document))
    
    p_theta0,p_theta1,probability_abusive = train_NB(np.array(train_mat), np.array(class_vectors))
    
    test_data = ['love', 'my', 'dalmation']
    test_word_vector = np.array(words_2_vector(my_vocabulary_list, test_data))
    print(test_data,'classified as: ',classify_NB(test_word_vector, p_theta0, p_theta1, probability_abusive))
    
    test_data = ['stupid', 'garbage']
    test_word_vector = np.array(words_2_vector(my_vocabulary_list, test_data))
    print(test_data,'classified as: ',classify_NB(test_word_vector, p_theta0, p_theta1, probability_abusive))

def main():
    testing_NB()

if __name__=='__main__':
    main()
