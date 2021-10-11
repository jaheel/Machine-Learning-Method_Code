# 集成学习

集成学习：构建并结合多个学习器来完成学习任务（多分类器系统）

结构：先产生一组“个体学习器”(individual learner)，再用某种策略将它们结合起来



同质(homogeneous)

> “决策树集成”中全是决策树，“神经网络集成”中全是神经网络
>
> 基学习器(base learner) ---> 基学习算法(base learning algorithm)



异质(heterogenous)

> 个体学习器由不同学习算法组成：组件学习器(component learner)



研究核心：如何产生并结合“好而不同”的个体学习器



分类：

1. 个体学习器间存在强依赖关系、必须串行生成的序列化方法

   > Boosting

2. 个体学习器间不存在强依赖关系、可同时生成的并行化方法

   > Bagging和“随机森林”(Random Forest)



## 1 Boosting

将弱学习器提升为强学习器的算法



代表算法：AdaBoost算法



## 2 Bagging

自助采样法(bootstrap sampling)

过程：给定m个样本的数据集，随机取一个再放回，重复m次，形成采样集；总共T个采样集，再基于采样集训练基学习器，再将这些基学习器结合。



样本扰动

##  3 随机森林(RF)

在以决策树为基学习器构建Bagging集成的基础上，进一步在决策树的训练过程中引入了随机属性选择。



样本扰动+属性扰动

## 4 结合策略

1. 平均法averaging（回归问题）

   > 简单平均法(Simple averaging)
   >
   > 加权平均法(weighted averaging)

2. 投票法 voting（分类问题）

   > 绝对多数投票法(majority voting)：必须占一半以上
   >
   > 相对多数投票法(plurality voting)：最多票数即可
   >
   > 加权投票法(weighted voting)

3. 学习法

   > Stacking