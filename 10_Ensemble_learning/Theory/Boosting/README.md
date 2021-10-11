# Boosting

原理：“三个臭皮匠顶个诸葛亮”



理论前置：

1. PAC学习框架中

   > 强可学习：一个概念（一个类），如果存在一个多项式的学习算法能够学习它，并且正确率很高。
   >
   > 弱可学习：一个概念（一个类），如果存在一个多项式的学习算法能够学习它，学习的正确率仅比随即猜测略好。

2. Schapire证明

   > 强可学习 与 弱可学习 是等价的。（在PAC框架下，一个概念是强可学习的充要条件是这个概念是弱可学习的。



问题：

已经发现了“弱学习算法”，那么能否将它提升(boost)为“强学习算法”？



提升方法：

​		从弱学习算法出发，反复学习，得到一系列弱分类器（又称为 基本分类器），然后组合这些弱分类器，构成一个强分类器。



对提升方法的两个问题：

1. 每一轮如何改变训练数据的权值或概率分布
2. 如何将弱分类器组合成一个强分类器



## AdaBoost

对问题的解答：

1. 提高那些被前一轮弱分类器错误分类样本的权值，而降低那些被正确分类样本的权值。

2. 弱分类器的组合（加权多数表决）

   > 加大分类误差率小的弱分类器的权值，使其在表决中起较大的作用
   >
   > 减小分类误差率大的弱分类器的权值，使其在表决中起较小的作用



算法：

1. 初始化训练数据的权值分布(保证第1步能够在原始数据上学习基本分类器$G_1(x)$)
   $$
   D_1 = (w_{11},\dotsb,w_{1i},\dotsb,w_{1N}), \space\space w_{1i}=\frac{1}{N}, \space \space i=1,2, \dotsb,N
   $$

2. 对$m=1,2,\dotsb,M$

   * 使用具有权值分布$D_m$的训练数据集学习，得到基本分类器
     $$
     G_m(x): \chi \to \{-1,+1\}
     $$

   * 计算$G_m(x)$在训练数据集上的分类误差率($w_{mi}$表示第$m$轮中第$i$个实例的权值)
     $$
     e_m = P(G_m(x_i) \ne y_i) = \sum_{i=1}^N w_{mi} I(G_m(x_i) \ne y_i) \\
     \sum_{i=1}^N w_{mi} =1
     $$

   * 计算$G_m(x)$的系数
     $$
     \alpha_m = \frac{1}{2} \log{\frac{1-e_m}{e_m}}
     $$

   * 更新训练数据集的权值分布
     $$
     D_{m+1}=(w_{m+1,1}, \dotsb, w_{m+1,i}, \dotsb, w_{m+1,N}) \\
     w_{m+1,i}=\frac{w_{mi}}{Z_m} \exp{(-\alpha_m y_i G_m(x_i))}, \space \space i=1,2,\dotsb,N \\
     Z_m = \sum_{i=1}^N {w_{mi} \exp{(-\alpha_m y_i G_m(x_i))}}
     $$
     使$D_{m+1}$成为一个概率分布

3. 构建基本分类器的线性组合
   $$
   f(x)=\sum_{m=1}^M \alpha_m G_m(x)
   $$
   得到最终分类器
   $$
   G(x)=sign(f(x))=sign(\sum_{m=1}^M \alpha_m G_m(x))
   $$



PS：

1. 基本分类器$G_m(x)$在加权训练数据集上的分类误差率：
   $$
   e_m = P(G_m(x_i) \ne y_i) = \sum_{G_m(x_i) \ne y_i} w_{mi}
   $$

2. $\alpha_m$表示$G_m(x)$在最终分类器中的重要性

   > 当$e_m \le \frac{1}{2}时, \alpha_m \ge 0$，并且$\alpha_m$随着$e_m$的减小而增大，所以分类误差率越小的基本分类器在最终分类器中的作用越大。

3. 更新训练数据的权值分布为下一轮作准备
   $$
   w_{m+1,i}= \begin{cases}
   \frac{w_{mi}}{Z_m} e^{-\alpha_m}, & G_m(x_i)=y_i \\
   \frac{w_{mi}}{Z_m} e^{\alpha_m}, & G_m(x_i) \ne y_i
   \end{cases}
   $$
   被基本分类器$G_m(x)$误分类样本的权值得以扩大，而被正确分类样本的权值却得以缩小。

4. 系数$\alpha_m$表示了基本分类器$G_m(x)$的重要性，所有$\alpha_m$之和并不为1。

5. $f(x)$的符号决定实例$x$的类，$f(x)$的绝对值表示分类的确信度。



## 前向分布算法

输入：训练数据集$T=\{(x_1,y_1),(x_2,y_2), \dotsb, (x_N,y_N) \}, \space  x_i \in \chi \sube R^n,, \space y_i \in Y=\{ -1,+1 \}$；损失函数$L(y,f(x))$；基函数集$\{b(x;\gamma\}$；

输出：加法模型$f(x)$

步骤：

1. 初始化$f_0(x)=0$

2. 对$m=1,2,\dotsb,M$

   * 极小化损失函数
     $$
     (\beta_m,\gamma_m) = \arg \min_{\beta,\gamma} \sum_{i=1}^N {L(y_i,f_{m-1}(x_i)+\beta b(x_i;\gamma))}
     $$
     得到参数$\beta_m,\gamma_m$

   * 更新
     $$
     f_m(x)=f_{m-1}(x)+\beta_m b(x;\gamma_m)
     $$

   * 得到加法模型
     $$
     f(x)=f_M(x)=\sum_{m=1}^M \beta_m b(x;\gamma_m)
     $$



## 提升树(GBDT Gradient Boosting Decision Tree)

提升树模型可以表示为决策树的加法模型：
$$
f_M(x)=\sum_{m=1}^M T(x;\varTheta_m)
$$
其中，$T(x;\varTheta_m)$表示决策树；$\varTheta_m$为决策树的参数；$M$为树的个数



回归问题的提升树算法：

输入：训练数据集$T=\{ (x_1,y_1),(x_2,y_2), \dotsb , (x_N,y_N) \} , \space x_i \in \chi \sube R^n, \space y_i \in Y \sube R$

输出：提升树$f_M(x)$

算法步骤：

1. 初始化$f_0(x)=0$

2. 对$m=1,2,\dotsb,M$

   * 计算残差
     $$
     r_{mi}=y_i - f_{m-1}(x_i), \space i=1,2,\dotsb,N
     $$

   * 拟合残差$r_{mi}$学习一个回归树，得到$T(x;\varTheta_m)$

   * 更新$f_m(x)=f_{m-1}(x)+T(x;\varTheta_m)$

3. 得到回归问题提升树
   $$
   f_M(x)=\sum_{m=1}^M T(x;\varTheta_m)
   $$