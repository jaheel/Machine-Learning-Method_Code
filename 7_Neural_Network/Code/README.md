# neuron network code practice

简单的实现一下神经网络

## Data Generate

神经网络原始数据的生成



## Data Preprocess

数据预处理





## Neuron

### Property

| name    | format                                  | explanation                  |
| ------- | --------------------------------------- | ---------------------------- |
| weights | [python_list]\(n_neurons_layer_before\) | 权值列表                     |
| bias    | 1_number                                | 偏置                         |
| result  | 1_number                                | 单个神经元训练结果（未激活） |
| error   | 1_number                                | 单个神经元误差               |



### Function

#### \__init__()

| initiate_parameter_name | format                                  | explanation |
| ----------------------- | --------------------------------------- | ----------- |
| weights                 | [python_list]\(n_neurons_layer_before\) |             |
| bias                    | 1_number                                |             |



#### update_weights()

| parameter_name | format                                  | explanation              |
| -------------- | --------------------------------------- | ------------------------ |
| delta          | [python_list]\(n_neurons_layer_before\) | 权值更新，变化幅度的记录 |

公式：
$$
w'_i = w_i + \Delta \\
\Delta = \mu * \delta_i * \frac{df_i(e)}{de} * x_i (在Layer层计算)
$$

#### feedforward()

| parameter_name | format                                                       | explanation |
| -------------- | ------------------------------------------------------------ | ----------- |
| input_data     | {array-like, np.array} of shape(1_samples, n_activate_result) | 输入数据    |

公式：
$$
y = \bold{W} \cdot \bold{X} + b
$$

| result_name | format   | explanation      |
| ----------- | -------- | ---------------- |
| result      | 1_number | 上述公式计算结果 |



## Layer

### Property

| name                  | format                                                       | explanation             |
| --------------------- | ------------------------------------------------------------ | ----------------------- |
| neuron_number_prev    | 1_number                                                     | 上一层神经元个数        |
| neuron_number_current | 1_number                                                     | 当前层神经元个数        |
| neurons               | {list-like, python_list} of shape(n_Neuron)                  | 神经元列表(python list) |
| activate_function_str | 1_str                                                        | 激活函数类型            |
|                       |                                                              |                         |
| input_data            | {array-like, np.array} of shape(1_samples, n_activate_result) | 输入数据                |
|                       |                                                              |                         |
| next                  | 指针                                                         | 指向后一个Layer         |
| prev                  | 指针                                                         | 指向前一个Layer         |



### Function

#### \__init__()

| parameter_name    | format                                                       | explanation                                              |
| ----------------- | ------------------------------------------------------------ | -------------------------------------------------------- |
| neuron_number     | {python_tuple} of shape(1_number_previous_layer_neurons, 1_number_current_layer_neurons) | 保存神经元个数的元组                                     |
| activate_function | 1_str                                                        | 1. relu(default) 2. sigmoid 3. tanh 4. leaky_relu 5. elu |



#### \__weight_init()

单个神经元权值初始化

方法：随机初始化

| parameter_name    | format   | explanation              |
| ----------------- | -------- | ------------------------ |
| in_neuron_numbers | 1_number | 与上一层神经元连接的个数 |



#### feedforward()

方法介绍：前馈网络计算，基于上一层的结果，计算这一层网络的结果(激活后)



| parameter_name | format                                                       | explanation |
| -------------- | ------------------------------------------------------------ | ----------- |
| input_data     | {array-like, np.array} of shape(1_samples, n_activate_result) | 输入数据    |



| result_name | format                                    | explanation         |
| ----------- | ----------------------------------------- | ------------------- |
| result      | [python list] of shape(n_activate_result) | n为当前层神经元个数 |



#### error_update()

方法介绍：更新该网络层每一个神经元的误差

| parameter_name    | format                                 | explanation                              |
| ----------------- | -------------------------------------- | ---------------------------------------- |
| error_update_list | [python list] of shape(n_neuron_error) | 该list已经保存了每个神经元，一一对应更新 |



## Network

### Property

| name        | format                                                   | explanation             |
| ----------- | -------------------------------------------------------- | ----------------------- |
| input_data  | {array-like, python list} of shape(1_sample, n_features) | 输入数据                |
| labels      | {array-like, python list} of shape(1_sample, n_results)  | 测试集结果              |
| _learn_rate | 1_number                                                 | BP algorithm learn rate |
|             |                                                          |                         |
| _head       | 指针                                                     | 指向第一层Layer         |
| _tail       | 指针                                                     | 指向最后一层Layer       |



### Function

#### \__init__()

| parameter_name | format                                                   | explanation        |
| -------------- | -------------------------------------------------------- | ------------------ |
| input_data     | {array-like, python list} of shape(1_sample, n_features) | 输入数据           |
| labels         | {array-like, python list} of shape(1_sample, n_results)  | 输入数据的监督结果 |
| learn_rate     | 1_number                                                 | BP算法 学习率      |



#### is_empty()

判断网络层是否为空



#### length()

计算网络层数

| result_name | format   | explanation  |
| ----------- | -------- | ------------ |
| count       | 1_number | 网络层数个数 |



#### items()

迭代返回layer

| yield_result_name | format  | explanation |
| ----------------- | ------- | ----------- |
| cur               | 1_layer | 返回当前层  |



#### add_layer()

往网络中添加layer

| parameter_name | format  | explanation |
| -------------- | ------- | ----------- |
| layer          | 1_layer | 单一网络层  |



#### predict()

约束条件：MSE错误率、迭代次数



前向传播、后向传播