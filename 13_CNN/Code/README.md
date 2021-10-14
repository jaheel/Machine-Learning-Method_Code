# 卷积神经网络

Convolutional Neural Network

卷积核(Convolution Kernel)类似视觉神经中的感受野

基本包括 卷积层、池化层、输出层



# 卷积层

## 卷积和卷积核

卷积核：也是一个矩阵，从数学上称为算子(option)，对输入矩阵进行变换。输入、输出的维数没变，但数值变了。



卷积定义：
$$
(I \otimes K)_{ij} = \sum_{m=0}^{k_1-1} \sum_{n=0}^{k_2-1} I(i+m, j+n) K(m, n)
$$

> 对应元素相乘，然后再求和



## 步长(strides)



步长(strides)：分为横向步长和纵向步长，代表一次卷积操作之后卷积核移动的距离



用C表示边长：
$$
C_{output}= \frac{C_{input} - C_{kernel}}{strides} + 1
$$


## 填充(padding)

填充(padding)：分为两种方式SAME和VALID

### VALID

$$
O_w = ceil \bigg{(} \frac{I_w - k_w + 1}{s_w} \bigg{)} \\
O_h = ceil \bigg{(} \frac{I_h - k_h + 1}{s_h} \bigg{)}
$$

超过$O_w, O_h$部分就舍弃不要了。

真正输入：
$$
I_w = s_w(O_w - 1) + k_w \\
I_h = s_h(O_h - 1) + k_h
$$

### SAME

在输入周围补0，补0后的输出大小：
$$
O_w = ceil \bigg{(} \frac{I_w}{s_w} \bigg{)} \\
O_h = ceil \bigg{(} \frac{I_h}{s_h} \bigg{)}
$$
根据应得到的输出大小进行padding操作：
$$
P_h = max((O_h - 1) \times s_h + k_h - I_h, 0) \\
P_w = max((O_w - 1) \times s_w + k_w - I_w, 0) \\
P_{top} = floor(\frac{P_h}{2}) \\
P_{bottom} = P_h - P_{top} \\
P_{left} = floor(\frac{P_w}{2}) \\
P_{right} = P_w - P_{left}
$$



## forward

前一层输出：

1. 定义好卷积核数目、大小、步长、填充方式，计算输出大小并进行padding，得到 输入$a^{l-1}$。

2. 初始化所有卷积和的权重$W$和偏置$b$。

3. 根据前向传播的公式(M个通道)：
   $$
   a^l = \sigma(z^l) = \sigma(\sum_{k=1}^M z_k^l) = \sigma (\sum_{k=1}^M a_k^{l-1} * W_k^l + b^l)
   $$
   即
   $$
   a^l = \sigma(z^l) = \sigma(a^{l-1} * W^l + b^l) \\
   *:卷积运算 \\
   \sigma() :激活函数
   $$
   



## backward

### 简化版

已知卷积层的$\delta^l$，通过反向传播计算上一层的$\delta^{l-1}$。

反向传播公式，链式法则：
$$
\delta^{l-1} = \frac{\partial J(W,b)}{\partial z^{l-1}} = \frac{\partial {J(W,b)}}{\partial {z^l}} \frac{\partial {z^l}}{\partial{z^{l-1}}} = \delta^l \frac{\partial {z^l}}{\partial{z^{l-1}}}
$$
又前向传播公式：
$$
a^l = \sigma(z^l) = \sigma(a^{l-1} * W^l + b^l)
$$
则有：
$$
z^l = \sigma(z^{l-1}) * W^l + b^l
$$
则有：
$$
\delta^{l-1} = \delta^l \frac{\partial z^l}{\partial z^{l-1}} = \delta^l * rot180(W^l) \odot \sigma'(z^{l-1})
$$

### 详细版

设网络的损失函数为：
$$
E = \frac{1}{2} \sum_p(t_p - y_p)^2
$$

$$
\begin{equation}
\begin{split}

\frac{\partial{E}}{\partial{w_{m',n'}^l}} &= \sum_{i=0}^{H-k_1} \sum_{j=0}^{W-k_2} \frac{\partial{E}}{\partial{x_{i,j}^l}} \frac{\partial{x_{i,j}^l}}{\partial{w_{m',n'}^l}}  \\
&= \sum_{i=0}^{H-k_1} \sum_{j=0}^{W-k_2} \delta_{i,j}^l \frac{\partial{x_{i,j}^l}}{\partial{w_{m',n'}^l}} \\

\frac{\partial{x_{i,j}^l}}{\partial{w_{m',n'}^l}} &= \frac{\partial}{\partial{w_{m',n'}^l}} \bigg{(} \sum_m \sum_n w_{m,n}^l o_{i+m,j+n}^{l-1} + b^l \bigg{)} \\
&= \frac{\partial}{\partial{w_{m',n'}^l}} (w_{0,0}^l o_{i+0,j+0}^{l-1} + \dots + w_{m',n'}^l o_{i+m',j+n'}^{l-1} + \dots + b^l) \\
&= \frac{\partial}{\partial{w_{m',n'}^l}} (w_{m',n'}^l o_{i+m',j+n'}^{l-1}) \\
&= o_{i+m',j+n'}^{l-1} \\

\frac{\partial{E}}{\partial{w_{m',n'}^l}} &= \sum_{i=0}^{H-k_1} \sum_{j=0}^{W-k_2} \delta_{i,j}^lo_{i+m',j+n'}^{l-1} \\
&= rot_{180°}\{ \delta_{i,j}^l \} * o_{m',n'}^{l-1} \\





\end{split}
\end{equation}
$$

$$
\begin{equation}
\begin{split}

\frac{\partial{E}}{\partial{x_{i',j'}^l} } &= \sum_{m=0}^{k_1-1} \sum_{n=0}^{k_2 - 1} \frac{ \partial{E}}{\partial{x_{i'-m,j'-n}^{l+1}}} \frac{\partial{x_{i'-m,j'-n}^{l+1}}}{ \partial{x_{i',j'}^l}} \\
&= \sum_{m=0}^{k_1-1} \sum_{n=0}^{k_2 - 1} \delta_{i'-m,j'-n}^{l+1} \frac{\partial{x_{i'-m,j'-n}^{l+1}}}{ \partial{x_{i',j'}^l}} \\

\frac{\partial{x_{i'-m,j'-n}^{l+1}}}{ \partial{x_{i',j'}^l}} &= \frac{\partial}{\partial{x_{i',j'}^l}} \bigg{(} \sum_{m'} \sum_{n'} w_{m',n'}^{l+1} o_{i'-m+m',j'-n+n'}^l + b^{l+1} \bigg{)} \\
&=  \frac{\partial}{\partial{x_{i',j'}^l}} \bigg{(} \sum_{m'} \sum_{n'} w_{m',n'}^{l+1} f(x_{i'-m+m',j'-n+n'}^l) + b^{l+1} \bigg{)} \\
&= \frac{\partial}{\partial{x_{i',j'}^l}} (w_{m'n'}^{l+1} f(x_{0-m+m',0-n+n'}^l) + \dots + w_{m,n}^{l+1} f(x_{i',j'}^l) + \dots + b^{l+1} ) \\
&= \frac{\partial}{\partial{x_{i',j'}^l}} (w_{m,n}^{l+1} f(x_{i',j'}^l)) \\
&= w_{m,n}^{l+1} \frac{\partial}{\partial{x_{i',j'}^l}}(f(x_{i',j'}^l)) = w_{m,n}^{l+1} f'(x_{i',j'}^l) \\

\frac{\partial E}{\partial{x_{i',j'}^l}} &= \sum_{m=0}^{k_1-1} \sum_{n=0}^{k_2-1} \delta_{i'-m,j'-n}^{l+1} w_{m,n}^{l+1} f'(x_{i',j'}^l) \\
&= rot_{180°} \bigg{(} \sum_{m=0}^{k_1-1} \sum_{n=0}^{k_2-1} \delta_{i'+m,j'+n}^{l+1} w_{m,n}^{l+1} \bigg{)} f'(x_{i',j'}^l) \\
&= \delta_{i',j'}^{l+1} * rot_{180°}\{ w_{m,n}^{l+1} \} f'(x_{i',j'}^l)

\end{split}
\end{equation}
$$

### 更新W，b

更新W：

求出了$\delta^l$的值，根据公式：
$$
z^l = a^{l-1} * W^l +b
$$
可求得：
$$
\frac{\partial{J(W,b)}}{\partial{W^l}} = \frac{\partial{J(W,b)}}{\partial{z^l}} \frac{\partial{z^l}}{\partial{W^l}} = a^{l-1} * \delta^l
$$


更新b：

对应卷积核的误差求和。

## Reference

[卷积神经网络CNN BP算法推导](https://www.cnblogs.com/chenjieyouge/p/12318116.html)

[卷积神经网络的Python实现](https://leonzhao.cn/2018/11/04/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%9A%84Python%E5%AE%9E%E7%8E%B0%EF%BC%88%E4%B8%89%EF%BC%89%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/)



# 池化层

作用：

* 减少计算量

  > 减少特征图尺寸，减少后面的层的计算量

* 提高多尺度信息

  > 如果存在多个池化层，就相当于网络中构造了一个多尺度特征金字塔，多尺度金字塔有利于提高检测/识别的稳定性

## forward

经过池化层后输出的高度和宽度分别为：
$$
H^l = (H^{l-1} + 2 \times p_h^{l-1} - k_h^{l-1}) / s_h^{l-1} + 1 \\
W^l = (W^{l-1} + 2 \times p_w^{l-1} - k_w^{l-1}) / s_w^{l-1} + 1
$$


最大池化的前向公式：
$$
z_{c,i,j}^l = \max_{i \cdot s_h^{l-1} \le m < i \cdot s_h^{l-1} + k_h^{l-1} \\ j \cdot s_w^{l-1} \le n < j \cdot s_w^{l-1} + k_w^{l-1}} (p z_{c,i,j}^{l-1}) \space \space \space \space \space i \in[0, H^l - 1], j \in [0, W^l - 1]
$$


平均池化的前向公式：
$$
z_{c,i,j}^l = \sum_{m = i \cdot s_h^{l-1}}^{i \cdot s_h^{l-1} + k_h^{l-1} - 1} \sum_{n = j \cdot s_w^{l-1}}^{j \cdot s_w^{l-1} + k_w^{l-1} - 1} (p z_{c,i,j}^{l-1})/(k_h^{l-1} \cdot k_w^{l-1}) \space \space \space \space \space i \in [0, H^l - 1], j \in [0, W^l - 1]
$$




## backward

由BP反向传播公式：
$$
\delta_{i}^{k-1} = \delta_{j}^k \nabla{f(a_j^k)} w_i^k
$$

* 池化层没有激励函数$f()$，可认为激励函数$f(x)=x$，其导数$\nabla f(x) = 1$
* 池化层没有可学习的权重$w$，可认为$w=1$



则上面的反向传播公式简化为：
$$
\delta_i^{k-1} = \delta_{j}^k
$$


1. mean pooling

   > mean pooling的前向传播是把一个patch中的值求取平均来做pooling
   >
   > 反向传播的过程就是把某个元素的梯度等分为n份分配给前一层
   >
   > PS ： 保证池化前后的梯度（残差）之和保持不变

2. max pooling

   > 把梯度传给前一层最大的那一个像素，其他像素不接受梯度，置为0



## Reference

[池化层的反向传播](https://blog.csdn.net/csuyzt/article/details/82633051?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_baidulandingword-6&spm=1001.2101.3001.4242)

[池化层反向传播公式推导](https://blog.csdn.net/z0n1l2/article/details/80892519?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-4.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-4.control)

[池化层的反向传播是怎么实现的](https://blog.csdn.net/qq_21190081/article/details/72871704)



# 全连接层

本质上就是最初的神经网络(NN)

## forward

前向传播公式：
$$
a^l = \sigma(z^l) = \sigma(a^{l-1}W^l + B^l )
$$
具体推导：
$$
z_{i}^l = \sum_{p=1}^{p_{l-1}} \bigg{(} a_{p}^{l-1} * w_{p,i}^l \bigg{)} + b_i^l \space \space \space i=1,2,\dots, K
$$


## backward

### MSE

输出层：
$$
E = \frac{1}{2}(t - a)^2 = \frac{1}{2} \sum_{i=1}^{K}(t_i - a_i)^2
$$
误差展开到隐层：
$$
E = \frac{1}{2} \sum_{i=1}^K \lbrack t_i - \sigma(z_i) \rbrack ^2
  = \frac{1}{2} \sum_{i=1}^K \lbrack t_i - \sigma(\sum_{j=0}^m    a_j w_{ji} ) \rbrack ^2
$$


针对当前层：
$$
\begin{equation}
\begin{split}

\frac{\partial E}{\partial{w_{pi}}}  &= \frac{\partial E}{\partial a_i} \frac{\partial a_i}{\partial z_i} \frac{\partial z_i}{\partial w_{pi}} \\
&= (a_i^l - t_i^l) \cdot \sigma'(z_i) \cdot a_p^{l-1} \\
&= \delta_i \cdot \sigma'(z_i) \cdot a_p^{l-1} \\

\frac{\partial E}{ \partial{b_{i}}} &= \frac{\partial E}{\partial a_i} \frac{\partial a_i}{\partial z_i} \frac{\partial z_i}{\partial b_{i}} \\
&= (a_i^l - t_i^l) \cdot \sigma'(z_i) \cdot 1 \\
& = \delta_i \cdot \sigma'(z_i)

\end{split}
\end{equation}
$$


### cross entropy error function

交叉熵损失函数（又称为softmax loss)：
$$
L = - \sum_{i=1}^n y_i \cdot \ln{(a_i)}
$$

初始输入：
$$
z = [z_1, z_2, \dots , z_n]
$$

经过softmax：
$$
a_i = \frac{e^{z_i}}{\sum_{k=1}^n e^{z_k}}
$$
链式法则：(a, z)均是(1,n)的向量
$$
\frac{\partial L}{\partial z} = \frac{\partial L}{\partial a} * \frac{\partial a}{\partial z}
$$
假设只有$y_j=1$，其余$y_i = 0$，则有：
$$
L = - y_j \ln(a_j) = - \ln{(a_j)}
$$

$$
\frac{\partial L}{\partial a} = [0,0,\dots,- \frac{1}{a_j},\dots,0]
$$

$$
\frac{\partial a}{\partial z} =
\begin{bmatrix}

\frac{\partial{a_1}}{\partial{z_1}} & \frac{\partial{a_1}}{\partial{z_2}} & \dots & \frac{\partial{a_1}}{\partial{z_n}} \\
\vdots & \vdots & & \vdots \\
\frac{\partial{a_j}}{\partial{z_1}} & \frac{\partial{a_j}}{\partial{z_2}} & \dots & \frac{\partial{a_j}}{\partial{z_n}} \\
\vdots & \vdots & & \vdots \\
\frac{\partial{a_n}}{\partial{z_1}} & \frac{\partial{a_n}}{\partial{z_2}} & \dots & \frac{\partial{a_n}}{\partial{z_n}}
\end{bmatrix}
$$

Jocobian矩阵每一行对应着$\frac{\partial{a_i}}{\partial{z}}$

由于$\frac{\partial L}{\partial a}$只有第j列不为0，由矩阵乘法，其实我们只要求$\frac{\partial a}{\partial z}$的第j行，也即$\frac{\partial a_j}{\partial z}$
$$
\frac{\partial L}{\partial z} = -\frac{1}{a_j} * \frac{\partial a_j}{\partial z}
$$
其中：
$$
a_j = \frac{e^{z_j}}{\sum_{k=1}^n e^{z_k}}
$$

* 当$i \neq j$时：
  $$
  \frac{a_j}{z_i} = \frac{0-e^{z_j}\cdot e^{z_i}}{(\sum_k^n e^{z_k})^2} = -a_j \cdot a_i \\
  \frac{\partial L}{\partial z_i} = -a_j \cdot a_i \cdot (- \frac{1}{a_j}) = a_i
  $$

* 当$i = j$时：
  $$
  \frac{a_j}{z_i} = \frac{e^{z_j} \cdot \sum_k^n e^{z_k} -e^{z_j}\cdot e^{z_i}}{(\sum_k^n e^{z_k})^2} = a_j - a_j^2 \\
  \frac{\partial L}{\partial z_i} = (a_j - a_j^2) \cdot (- \frac{1}{a_j}) = a_j - 1
  $$



从而推出：
$$
\frac{\partial L}{\partial z} = [a_1,a_2,\dots,a_j-1,\dots,a_n] = a-y
$$


## Reference

[15分钟搞定Softmax Loss求导](https://zhuanlan.zhihu.com/p/105758059)

[反向传播算法”过程及公式推导](https://blog.csdn.net/ft_sunshine/article/details/90221691)