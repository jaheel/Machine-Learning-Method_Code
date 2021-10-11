# Bagging

## algorithm

bootstrap aggregating

算法过程：

1. 从原始样本集中抽取训练集。

   > 每轮从原始样本集中使用Bootstraping方法抽取n个训练样本。共进行k轮，形成k个训练集(k个训练集之间是相互独立的)

2. 每次使用一个训练集得到一个模型，k个训练集共得到k个模型

   > 并没有具体的分类算法或回归算法，可以根据具体问题采用不同的分类或回归方法，如决策树、感知器等

3. 模型预测

   > 分类问题：上步得到的k个模型采用投票的方式得到分类结果
   >
   > 回归问题：计算上述模型的均值作为最后的结果



## Contract with Boosting

1. 样本选择

   > Bagging：训练集自助采样，选出的各轮训练集之间互相独立
   >
   > Boosting：每一轮训练集不变，训练集每个样例在分类器中的权重发生变化。权值是根据上一轮的分类结果进行调整。

2. 样例权重

   > Bagging：使用均匀采样，每个样例的权重相等
   >
   > Boosting：根据错误率不断调整样例的权值，错误率越大则权重越大

3. 预测函数

   > Bagging：权重相等
   >
   > Boosting：每个弱分类器都有对应的权重

4. 并行计算

   > Bagging：各个预测函数可并行完成
   >
   > Boosting：各个预测函数顺序生成，后一个模型参数需要前一轮模型的结果



# Random Forest

* Bagging的一个扩展变体

* 以决策树为基学习器构建Bagging集成的基础上，进一步在决策树的训练过程中引入了随机属性选择

* 传统决策树在选择划分属性时是在当前结点的属性集合（假定有d个属性）中选择一个最优属性；而在RF中，对基决策树的每个结点，先从该结点的属性集合中随机选择一个包含k个属性的子集，然后再从这个子集中选择一个最优属性用于划分。

  > k=d，则基决策树的构建与传统决策树相同
  >
  > k=1，则随机选择一个属性用于划分
  >
  > $k=\log_2d$，最佳

  

步骤：

1. 随机抽样，训练决策树
2. 随机选取属性，做结点分裂属性
3. 重复步骤2，直到不能再分裂
4. 建立大量决策树形成森林