# 聚类(clustering)

将数据集中的样本划分为若干个通常是不相交的子集，每个子集称为一个“簇”(cluster)。

聚类既能作为一个单独过程，用于找寻数据内在的分布结构，也可作为分类等其他学习任务的前驱过程。

## 1 性能度量

有效性指标(validity index)



簇内相似度(intra-cluster similarity)高

簇间相似度(inter-cluster similarity)低



外部指标(external index)

> 聚类结果与某个”参考模型“(reference model)进行比较
>
> 1. Jaccard系数(JC)
>    $$
>    JC=\frac{a}{a+b+c}
>    $$
>
> 2. FM指数(FMI)
>    $$
>    FMI=\sqrt{\frac{a}{a+b}\cdot \frac{a}{a+c}}
>    $$
>
> 3. Rand指数(RI)
>    $$
>    RI=\frac{2(a+b)}{m(m-1)}
>    $$

内部指标(internal index)

> 直接考察聚类结果而不利用任何参考模型
>
> 1. DB指数(DBI)：越小越好
>    $$
>    DBI=\frac{1}{k}\sum^{k}_{i=1} {max}_{j \neq i}(\frac{avg(C_i)+avg(C_j)}{d_{cen}{({\mu}_i},{\mu}_j)})
>    $$
>
> 2. Dunn指数(DI)：值越大越好
>    $$
>    DI={min}_{1\leq i \leq k}{\{ {min}_{j \neq i} ( \frac{d_{min}(C_i,C_j)}{max_{1 \leq l \leq k} diam(C_l)} ) \}}
>    $$
>    

## 2 距离计算

闵可夫斯基距离(Minkowski distance)
$$
dist_{mk}(x_i,x_j)=(\sum^{n}_{u=1}{|x_{iu}-x_{ju}|^p})^{\frac{1}{p}}
$$
p=2时，欧氏距离(Euclidean distance)
$$
dist_{ed}(x_i,x_j)=||x_i-x_j||_2 = \sqrt{\sum_{u=1}^{n} |x_{iu}-x_{ju}|^2}
$$
p=1时，曼哈顿距离(Manhattan distance)
$$
dist_{man}(x_i,x_j)=||x_i-x_j||_1 = \sum_{u=1}^{n}|x_{iu}-x_{ju}|
$$

## 3 原型聚类

k-均值

学习向量量化(Learning Vector Quantization)

高斯混合聚类(Mixture-of-Gaussian)