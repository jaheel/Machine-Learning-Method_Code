# Computational learning theory

## Basic information

研究的是关于通过“计算”来进行”学习“的理论，即关于机器学习的理论基础，其目的是分析学习任务的困难本质，为学习算法提供理论保证，并根据分析结果知道算法设计。

令$h$为从$X$到$y$的一个映射，其泛化误差为：
$$
E(h;D) = P_{x \backsim D}(h(x) \neq y)
$$
$h$在$D$上的经验误差为：
$$
\hat{E}(h;D) = \frac{1}{m} \sum_{i=1}^m (h(x_i) \neq y_i)
$$
令$\epsilon$为$E(h)$的上限，即$E(h) \leq \epsilon$ ; 我么通常用$\epsilon$表示预先设定的学得模型所应满足的误差要求，亦称”误差参数“。







若$h$在数据集$D$上的经验误差为0，则称$h$与$D$一致，否则称其与$D$不一致。对任意两个映射$h_1,h_2 \in X \rightarrow y$，可通过其”不合“(disagreement)来度量它们之间的差别：
$$
d(h_1,h_2) = P_{x \backsim D}(h_1(x) \neq h_2(x))
$$


* Jensen不等式：对任意凸函数$f(x)$，有：
  $$
  f(\mathbb{E}(x)) \leq \mathbb{E}(f(x))
  $$

* Hoeffding不等式[Hoeffding, 1963]：若$x_1,x_2,...,x_m$为m个独立随机变量，且满足$0 \leq x_i \leq 1$，则对任意 $\epsilon > 0$，有：
  $$
  P(\frac{1}{m} \sum_{i=1}^m x_i -\frac{1}{m} \sum_{i=1}^m \mathbb{E}(x_i) \geq \epsilon ) \leq \exp(-2m \epsilon^2) \\
  P(|\frac{1}{m} \sum_{i=1}^m x_i -\frac{1}{m} \sum_{i=1}^m \mathbb{E}(x_i) |\geq \epsilon) \leq 2\exp(-2m \epsilon^2)
  $$

* McDiarmid不等式[McDiarmid, 1989]：若$x_1,x_2,...,x_m$为m个独立随机变量，且对任意$1 \leq i \leq m$，函数$f$满足：
  $$
  \sup_{x_1,...,x_m} |f(x_1,...,x_m)-f(x_1,...,x_{i-1},x_i' ,x_{i+1},...,x_m)| \leq c_i
  $$
  则对任意$\epsilon > 0$，有：
  $$
  P(f(x_1,...,x_m)- E(f(x_1,...,x_m)) \ge \epsilon) \leq \exp(\frac{-2 \epsilon^2}{\sum_i c_i^2}) \\
  P(|f(x_1,...,x_m)- E(f(x_1,...,x_m))| \ge \epsilon) \leq 2\exp(\frac{-2 \epsilon^2}{\sum_i c_i^2})
  $$
  

## PAC

Probably Approximately Correct learning theory



令$c$表示”概念“(concept)，这是从样本空间$X$到标记空间$y$的映射，它决定示例$x$的真实标记$y$，若对任何样例$(x,y)$有$c(x)=y$成立，则称$c$为**目标概念**

目标概念的集合称为"概念类"(concept class)，用符号$C$表示



给定学习算法$\xi$，它所考虑的所有可能概念的集合称为”假设空间“(hypothesis space)，用符号$H$表示。(学习算法事先并不知道概念类的真实存在，因此$H$与$C$通常是不同的，学习算法会把自认为可能的目标概念集中起来构成$H$，对$h \in H$，由于并不能确定它是否真是目标概念，因此称为”假设“(hypothesis)。假设$h$也是从样本空间$X$到标记空间$y$的映射)



目标概念$c \in H$，称该问题对学习算法$\xi$是”可分的“(separable)，亦称”一致的“(consistent)

若$c \notin H$，称该问题对学习算法$\xi$是”不可分的“(non-separable)，亦称”不一致的“(non-consistent)





* PAC Identify（PAC辨识）

  > 对$0<\epsilon ,\delta <1$, 所有$c \in C$和分布D，若存在学习算法$\xi$ , 其输出假设$h \in H$ 满足
  > $$
  > P(E(h) \leq \epsilon) \geq 1-\delta
  > $$
  > 则称学习算法$\xi$能从假设空间$H$中PAC辨识概念类$C$.

* PAC Learnable（PAC可学习）

  > 令$m$表示从分布$D$中独立同分布采样得到的样例数目，$0< \epsilon ,\delta < 1$,对所有分布$D$, 若存在学习算法$\xi$和多项式函数$poly(\cdot, \cdot, \cdot, \cdot)$，使得对于任何$m \ge poly(1/ \epsilon, 1/ \delta, size(x),size(c))$，$\xi$能从假设空间$H$中PAC辨识概念类$C$，则称概念类$C$对假设空间$H$而言是PAC可学习的(简称：概念类C是PAC可学习的)

* PAC Learning Algorithm（PAC学习算法）

  > 若学习算法$\xi$使概念类$C$为PAC可学习的，且$\xi$的运行时间也是多项式函数$poly(1/ \epsilon, 1/ \delta, size(x),size(c))$，则称概念类$C$使高效PAC可学习(efficiently PAC learnable)。（$\xi$为概念类$C$的PAC学习算法）

* Sample Complexity（样本复杂度）

  > 满足PAC学习算法$\xi$所需的$m \ge poly(1/ \epsilon, 1/ \delta, size(x), size(c))$中最小的$m$，称为学习算法$\xi$的样本复杂度





一般而言，研究 假设空间 与 概念类 不同的情形，即$H \ne C$

$H$越大，其包含任意 目标概念 的可能性越大，但从中找到某个具体 目标概念 的难度也越大。

$|H|$有限时，称$H$为”有限假设空间“，否则称为”无限假设空间“



## finite hypothesis space

### separable

意味着：目标概念$c$属于假设空间$H$，即$c \in H$. 给定包含$m$个样例的训练集$D$，如何找出满足误差参数的假设呢？



需要多少样例才能学得目标概念$c$的有效近似呢？

> 对PAC学习来说，只要训练集D的规模能使学习算法$\xi$以概率$1-\delta$找到目标假设的$\delta$近似即可。



假定$h$泛化误差大于$\epsilon$，对分布$D$上随机采样而得的任何样例$(x,y)$有：
$$
\begin{equation}
\begin{split}
P(h(x)=y) &= 1-P(h(x) \neq y)\\
&= 1-E(h) \\
&< 1- \epsilon
\end{split}
\end{equation}
$$
由于D包含m个独立同分布采样而得的样例，因此，$h$与$D$表现一致的概率为：
$$
\begin{equation}
\begin{split}
P((h(x_1)=y_1) \land ... \land(h(x_m) = y_m)) &= (1-P(h(x) \neq y))^m \\
&< (1- \epsilon)^m

\end{split}
\end{equation}
$$
事先并不知道学习算法$\xi$会输出$H$中的哪个假设，但仅需保证泛化误差大于$\epsilon$，且在训练集上表现完美的所有假设出现概率之和不大于$\delta$即可：
$$
\begin{equation}
\begin{split}
P(h \in H: E(h) > \epsilon \land \hat{E}(h)=0) &< |H|(1- \epsilon)^m \\
&<|H|e^{-m \epsilon}
\end{split}
\end{equation}
$$
令该式不大于$\delta$，即
$$
|H|e^{-m \epsilon} \leq \delta
$$
可得：
$$
m \geq \frac{1}{\epsilon} (\ln{|H|} + \ln{\frac{1}{\delta}})
$$
可知：

> 有限假设空间$H$都是PAC可学习的，所需样例数目如上所示，输出假设$h$的泛化误差随着样例数目的增多而收敛到0，收敛速度为$O(\frac{1}{m})$

### non-separable

对较为困难的学习问题，目标概念$c$往往不存在于假设空间$H$中。假定对于任何$h \in H, \hat{E}(h) \neq 0$（$H$中的任意一个假设都会在训练集上出现或多或少的错误。



### agnostic PAC learnable

不可知PAC可学习

> 令$m$表示从分布$D$中独立同分布采样得到的样例数目，$0< \epsilon,\delta <1$，对所有分布$D$，若存在学习算法$\xi$和多项式函数$poly(\cdot,\cdot,\cdot,\cdot)$，使得对于任何$m \ge poly(1/ \epsilon, 1/ \delta, size(x),size(c))$，$\xi$能从假设空间$H$中输出满足下式的假设$h$：
> $$
> P(E(h)-\min_{h' \in H}E(h') \le \epsilon) \ge 1- \delta
> $$
> 则称假设空间$H$是不可知PAC可学习的。