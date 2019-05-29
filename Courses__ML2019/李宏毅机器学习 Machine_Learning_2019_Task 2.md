# 李宏毅机器学习 Machine_Learning_2019_Task2

## 机器学习打卡任务内容：

### 理解偏差 (Bias) & 方差 (Variance)

- 偏差（**bias**）和方差（**variance**）的含义
  
  - 泛化误差可以分解成偏差的平方加上方差加上噪声。偏差度量了学习算法的期望预测和真实结果的偏离程度,刻画了学习算法本身的拟合能力,方差度量了同样大小的训练集的变动所导致的学习性能的变化,刻画了数据扰动所造成的影响,噪声表达了当前任务上任何学习算法所能达到的期望泛化误差下界,刻画了问题本身的难度；
  - 偏差和方差一般称为bias和variance,一般训练程度越强,偏差越小,方差越大,泛化误差一般在中间有一个最小值,如果偏差较大,方差较小,此时一般称为欠拟合,而偏差较小,方差较大称为过拟合； 
  - 偏差（bias）和方差（variance）的常见解释
    - 解释1
      bias 偏差 ：模型的期望（或平均）预测和正确值之间的差别；
      variance 方差 ：模型之间的多个拟合预测之间的偏离程度。
    - 解释2：
      bias和variance分别从两个方面来描述了我们学习到的模型与真实模型之间的差距；
      bias是 “用所有可能的训练数据集训练出的所有模型的输出的平均值” 与 “真实模型”的输出值之间的差异；
      variance则是“不同的训练数据集训练出的模型”的输出值之间的差异。
    - 解释3：
      首先 Error = bias + variance
      Error反映的是整个模型的准确度，bias反映的是模型在样本上的输出与真实值之间的误差，即模型本身的精准度，variance反映的是模型每一次输出结果与模型输出期望之间的误差，即模型的稳定性；
      更准确地讲Error分成3个部分：Error = bias + variance + noise;
  - 解决bias和variance问题的方法：
    - 高偏差，解决方案: Boosting、复杂模型(非线性模型、增加神经网络中的层)、增加特征 等；
    - 高方差，解决方案: Bagging、简化模型、降维等。
  
- 如何根据偏差**bias**和方差**variance**对模型进行分析和改善
  
  - 理想情况下，我们希望得到一个偏差和方差都很小的模型，但实际上往往很困难；
  
  - 模型评估方法：模型训练的时候使用的数据集是训练集，模型在测试集上的误差近似为泛化误差，而我们更关注的就是泛化误差，所以在 Off-line Phase 我们需要解决一个问题，那就是如何将一个数据集划分成训练集和测试集；
    
    - 留出法Hold-out：
    
      - 是指将数据集 D 划分成两份互斥的数据集，一份作为训练集 S，一份作为测试集 T，在 S 上训练模型，在 T 上评估模型效果；
      - 尽量保证训练集 S 和测试集 T 的数据分布一致，避免由于数据划分引入额外的偏差而对最终结果产生影响.
    
    - 交叉验证法Cross-Validation：
    
      ![5-CV](C:\Users\FengZhang\AppData\Roaming\Typora\typora-user-images\1557992901599.png)
    
      ​																				**5折交叉验证**
    
      - 先将数据集 D 划分成 k 分互斥的数据子集，即
        $$
      D=D_{1} \cup D 2 \cup \ldots \cup D_{k}.
        $$
        一般每个数据子集的个数基本相近、数据分布基本一致；然后每次用一份数据子集作为测试集，其余的 k-1 份数据子集作为训练集，迭代 k 轮得到 k 个模型，最后将将 k 次的评估结果汇总求平均值得到最终的评估结果
    
  - 自助法Bootstrapping：
    
      - 以自主采样（bootstrap sampling）为基础，使用有放回的重复采样的方式进行训练集、测试集的构建。比如为了构建 m 条样本的训练集，每次从数据集 D 中采样放入训练集，然后有放回重新采样，重复 m 次得到 m 条样本的训练集，然后将将没有出现过的样本作为测试集
    
    - 参数调节（调参）
    
  - 性能度量
    
    - 错误率与精度
    - 查准率、查全率与F1-Score
    - ROC与AUC
    
  - 通过实验估计学习算法的泛化性能，同时也可以通过“偏差-方差分解”来解释学习算法的泛化性能；偏差度量了学习算法的期望预测与真实结果的偏离程度，即刻画了学习算法本身的拟合能力；方差度量了同样大小的训练集的变动所导致的学习性能的变化，即刻画了数据扰动所造成的影响；噪声则表达了在当前任务上任何学习算法所能达到的期望泛化误差的下界，即刻画了学习问题本身的难度；
  
  - “偏差-方差分解”说明，泛化性能是由学习算法的能力、数据的充分性以及学习任务本身的难度所共同决定的。给定学习任务，为了取得好的泛化性能，则需使偏差较小，即能够充分拟合数据，并且使方差较小，即使得数据扰动产生的影响小；
  
  - 一般来说，偏差与方差是有冲突的，这称为“偏差-方差窘境”。假定我们能控制学习算法的训练程度，则在训练不足时，学习器的拟合能力不够强，训练数据的扰动不足以便学习器产生显著变化，此时“偏差”主导了泛化错误率;随着训练程度的加深，学习器的拟合能力逐渐增强，训练数据发生的扰动渐渐能被学习器学到，“方差”逐渐主导了泛化错误率;在训练程度充足后，学习器的拟合能力已非常强，训练数据发生的轻微扰动都会导致学习器发生显著变化，若训练数据自身的、非全局的特性被学习器学到了，则将发生“过拟合”。
    
    - 注意点：
      - 我们通常把学习得到的模型在实际使用中所遇到的数据称为“测试数据”！
      - 模型评估与选择中用于评估测试的数据集称为“验证集”！
      - 在研究对比不同算法的泛化性能时，我们常用“测试集”上的判别效果来估计模型在实际使用时的泛化能力，而把“训练数据”另外划分为“训练集”和“验证集”，基于验证集上的性能来进行模型选择和调参！
  
- 学习误差 (**Error**) 为什么是偏差和方差而产生的，并且推导数学公式

  - 一般而言，在测试数据上的误差由偏差(Bias)和方差(Variance)两部分组成；

    ![Bias-Variance](C:\Users\FengZhang\AppData\Roaming\Typora\typora-user-images\1557934264005.png)

    ​																	**偏差 & 方差 (Bias & Variance)**

    ![Bias-Variance](C:\Users\FengZhang\AppData\Roaming\Typora\typora-user-images\1557934598620.png)

    ​												  **偏差、方差与模型复杂度 (Bias-Variance & MC)**

    ![Bias&Variance](C:\Users\FengZhang\AppData\Roaming\Typora\typora-user-images\1557935122788.png)

    ​																			**Bias&Variance**

    ![bias&variance](C:\Users\FengZhang\AppData\Roaming\Typora\typora-user-images\1557934813026.png)

    ​																**预测值、期望值、真实值、偏差与方差**

  - 我们将拟合一个线性假设
    $$
    h(\mathrm{x})=\mathbf{w}^{T} \mathbf{x}
    $$
    在训练数据中，最小化平方和误差
    $$
    \sum_{i=1}^{m}\left(y_{i}-h\left(\mathbf{x}_{i}\right)\right)^{2}
    $$
    其中，
    $$
    {样本:}\langle\mathrm{x}, y\rangle
    $$

    $$
    y=f(\mathrm{x})+\epsilon
    $$

    $$
    \epsilon 是均值为0，\text {标准差为} \sigma 的高斯噪声
    $$

    假设样本之间是相互独立同分布的，即
    $$
    P(\langle\mathrm{x}, y\rangle)=P(\mathrm{x}) P(y | \mathrm{x})
    $$
    给定一个任意的样本点**x**，目的是分析和计算
    $$
    E_{P}\left[(y-h(\mathrm{x}))^{2} | \mathrm{x}\right]
    $$
    对于给定的假设类，我们还可以计算真实误差，即输入分布上的期望误差
    $$
    \sum_{\mathbf{x}} E_{P}\left[(y-h(\mathbf{x}))^{2} | \mathbf{x}\right] P(\mathbf{x})
    $$
    如果**x**为连续变量，则求和将变为积分形式；我们将把这个期望进行分解：

    **回顾期望与方差的推导过程**：

    - **X**为一随机变量，对应的概率分布为**P**(**X**)，**X**的均值 or 期望表示为

    $$
    E[X]=\sum_{i=1}^{n} x_{i} P\left(x_{i}\right)
    $$

    $$
    \begin{aligned} \operatorname{X的方差为：Var}[X] &=E\left[(X-E(X))^{2}\right] \\ &=E\left[X^{2}\right]-(E[X])^{2} \end{aligned}
    $$

    $$
    \begin{aligned} \operatorname{Var}[X] &=E\left[(X-E[X])^{2}\right] \\ &=\sum_{i=1}^{n}\left(x_{i}-E[X]\right)^{2} P\left(x_{i}\right) \\ &=\sum_{i=1}^{n}\left(x_{i}^{2}-2 x_{i} E[X]+(E[X])^{2}\right) P\left(x_{i}\right) \\ &=\sum_{i=1}^{n} x_{i}^{2} P\left(x_{i}\right)-2 E[X] \sum_{i=1}^{n} x_{i} P\left(x_{i}\right)+(E[X])^{2} \sum_{i=1}^{n} P\left(x_{i}\right) \\ &=E\left[X^{2}\right]-(E[X])^{2} \\ &=E\left[X^{2}\right]-(E[X])^{2} \end{aligned}
    $$

    $$
    E\left[X^{2}\right]=(E[X])^{2}+\operatorname{Var}[X]
    $$

    $$
    \begin{aligned} E_{P}\left[(y-h(\mathbf{x}))^{2} | \mathbf{x}\right] &=E_{P}\left[(h(\mathbf{x}))^{2}-2 y h(\mathbf{x})+y^{2} | \mathbf{x}\right] \\ &=E_{P}\left[(h(\mathbf{x}))^{2} | \mathbf{x}\right]+E_{P}\left[y^{2} | \mathbf{x}\right]-2 E_{P}[y | \mathbf{x}] E_{P}[h(\mathbf{x}) | \mathbf{x}]{......}(1) \end{aligned}
    $$

    $$
    令：\overline{h}(\mathrm{x})=E_{P}[h(\mathrm{x}) | \mathrm{x}]{，即在x处假设的均值预测}
    $$

    $$
    (1)式的第一项有：E_{P}\left[(h(\mathrm{x}))^{2} | \mathrm{x}\right]=E_{P}\left[(h(\mathrm{x})-\overline{h}(\mathrm{x}))^{2} | \mathrm{x}\right]+(\overline{h}(\mathrm{x}))^{2}
    $$

    $$
    定义：E_{P}[y | \mathrm{x}]=E_{P}[f(\mathrm{x})+\epsilon | \mathrm{x}]=f(\mathrm{x}){......(2)}
    $$

    $$
    (2)式的转化是因为：\epsilon \sim \mathcal{N}(0, \sigma) )
    $$

    
    $$
    有：E\left[y^{2} | \mathrm{x}\right]=E\left[(y-f(\mathrm{x}))^{2} | \mathrm{x}\right]+(f(\mathrm{x}))^{2}
    $$

    - 综上：
      $$
      \begin{aligned} E_{P}\left[(y-h(\mathrm{x}))^{2} | \mathrm{x}\right] &=E_{P}\left[(h(\mathrm{x})-\overline{h}(\mathrm{x}))^{2} | \mathrm{x}\right]+(\overline{h}(\mathrm{x}))^{2}-2 f(\mathrm{x}) \overline{h}(\mathrm{x}) \\ &+E_{P}\left[(y-f(\mathrm{x}))^{2} | \mathrm{x}\right]+(f(\mathrm{x}))^{2} \\ &=E_{P}\left[(h(\mathrm{x})-\overline{h}(\mathrm{x}))^{2} | \mathrm{x}\right]+(f(\mathrm{x})-\overline{h}(\mathrm{x}))^{2} \\ &+E\left[(y-f(\mathrm{x}))^{2} | \mathrm{x}\right] \end{aligned}
      $$
      其中，第一项表明从P中随机抽取有限数据集进行训练时，假设h在x处的方差：
      $$
      E_{P}\left[(h(\mathrm{x})-\overline{h}(\mathrm{x}))^{2} | \mathrm{x}\right]
      $$
      第二项为平方偏差 or 系统误差，它与我们考虑的类别假设有关：
      $$
      (f(\mathbf{x})-\overline{h}(\mathbf{x}))^{2}
      $$
      最后一项是噪音，是由于我们人为因素不可避免的误差：
      $$
      E\left[(y-f(\mathbf{x}))^{2} | \mathbf{x}\right]
      $$
      ![误差分解](C:\Users\FengZhang\AppData\Roaming\Typora\typora-user-images\1557934089989.png)

      ​																				   **误差分解**

- 过拟合，欠拟合，分别对应**Bias**和**Variance**什么情况？

  - 过拟合：简单地理解过拟合就是模型过分学习并拟合数据导致模型泛化性能较差；通过正则化的方法，可以尽量避免过拟合的发生。

  - 欠拟合：简单地理解就是数据的拟合程度过于简单，以至于无法获取训练数据之间的关系.

    ![Overfitting&Underfitting](C:\Users\FengZhang\AppData\Roaming\Typora\typora-user-images\1557935826468.png)

    ​																	**Overfitting & Underfitting**

- 偏差 & 方差之间的权衡 (机器学习的主要挑战问题之一)
  - 统计学习与机器学习的一个重要理论结果是，模型的泛化误差可以表示为三个不同的误差之和，即偏差 (**Bias**)、方差 (**Variance**)、以及不可约误差 (**Irreducible Error**).
    - **偏差** (Bias)：这部分泛化误差是由于错误的假设造成的，比如假设数据是线性的，而实际上它是二次的。高偏差模型最有可能在训练数据中产生欠拟合，即所谓的训练数据欠拟合；
    - 方**差** (Variance)：这部分是由于模型对训练数据的微小变化过于敏感。具有多个自由度的模型(比如，高次多项式模型)可能存在较大的方差，从而使训练数据过度拟合，即所谓的训练数据的过拟合；
    - **不可约误差** (Irreducible Error)：这部分是由于数据本身的噪声。减少这部分误差的唯一方法是清理数据(比如，修复数据源，如损坏的传感器，或检测并删除异常值).
  - **增加模型的复杂度**通常会**增加**其**方差**并**减少**其**偏差**。相反，**降低模型的复杂度**会**增加**其**偏差**并**降低**其**方差**。最终的目的就是为了达到偏差与方差之间的权衡.
  - 机器学习的主要挑战
    - 不充分的训练数据
    - 不具代表性的训练数据
    - 质量较差的数据
    - 无关联特征

### 学习鞍点，复习 Task 1 的全局最优 & 局部最优

- 鞍点 (Saddle Point)
  - What
    - 什么是鞍点 (形象记忆马鞍面上的点)
      - 简言之，一个`不是`**局部极值点**的**驻点**称为**鞍点**
    - 什么是驻点
      - 函数在一点处的**一阶导数**为零
    - 什么是 **Hessian** 矩阵
  - How
    - 如何证明一个点为鞍点
  - Comparison
    - 局部最小值与鞍点的区别

- 解决办法有哪些？
  - 一般而言，如果线性规划问题有最优解，那么其最优解必定可以在可行域的极点上达到；如果只有唯一最优解，则必定在极点上达到，而非线性规划的最优解可能是可行域的热河一点；
  - 线性规划的最优解一定是全局最优解，而非线性规划有局部最优解和全局最优解之分，一般的非线性规划算法往往求出的是局部最优解；
  - 可以利用凸函数的性质求解问题的极小值点；
  - 区间收缩法；
  - 加步探索法；
  - 牛顿法；
  - 抛物线法.
### 梯度下降法 (Gradient Descent)

> 梯度下降法是一种非常通用的优化算法，它能够为各种问题找到最优解。梯度下降的一般思想是迭代地调整参数，以使成本函数最小化；同时，当使用梯度下降时，您应该确保所有的特征都具有相似的比例。
>
> ![Gradient Descent](file:///C:/Users/FengZhang/Desktop/2019-05-16_100058.png)
>
> ​																				**梯度下降法**
>
> ![local minimum](C:\Users\FengZhang\AppData\Roaming\Typora\typora-user-images\1557998042523.png)
>
> ​																			**容易陷入局部最小解**

- 梯度下降法 Gradient Descent  对比记忆 梯度上升 Gradient Ascent

- 批量梯度下降法 Batch Gradient Descent

  > 利用代价函数的偏导数，求出梯度向量
  > $$
  > \frac{\partial}{\partial \theta_{j}} \operatorname{MSE}(\boldsymbol{\theta})=\frac{2}{m} \sum_{i=1}^{m}\left(\boldsymbol{\theta}^{T} \mathbf{x}^{(i)}-y^{(i)}\right) x_{j}^{(i)}
  > $$
  > 其中，成本 or 代价函数的梯度向量为
  > $$
  > \nabla_{\theta} \operatorname{MSE}(\theta)=\left( \begin{array}{c}{\frac{\partial}{\partial \theta_{0}} \operatorname{MSE}(\theta)} \\ {\frac{\partial}{\partial \theta_{1}} \operatorname{MSE}(\theta)} \\ {\vdots} \\ {\frac{\partial}{\partial \theta_{n}} \operatorname{MSE}(\theta)}\end{array}\right)=\frac{2}{m} X^{T}(X \theta-y)
  > $$
  > 梯度步长满足
  > $$
  > \theta^{(\text { next step })}=\theta-\eta \nabla_{\theta} \operatorname{MSE}(\theta)
  > $$

  `一个简单的算法实现`

  ```python
  eta = 0.1 # 学习率
  n_iterations = 1000
  m = 100
  
  theta = np.random.randn(2,1) # 随机初始化
  
  for iteration in range(n_iterations):
      gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
      theta = theta - eta * gradients
  ```

- 随机梯度下降法 Stochastic Gradient Descent

  > 批量梯度下降的主要问题是，它使用整个训练集来计算每一步的梯度，这使得当训练集很大时，计算速度非常慢。在相反的极端情况下，随机梯度下降只是在训练集中每一步选择一个随机实例，并仅基于该实例计算梯度。显然，这使得算法更快，因为它在每次迭代中只有很少的数据可以操作。它还使得在大型训练集上进行训练成为可能，因为在每次迭代中只需要在内存中存储一个实例(SGD可以作为一种**out-of-core**算法来实现)；
  >
  > 同时，由于其随机性(即与批量梯度下降相比，这种算法的规则性要差得多：成本函数不是缓慢下降，直到达到最小值，而是上下跳动，平均只下降。随着时间的推移，它将非常接近于最小值，但其一旦到达，它将继续反弹，永远不会稳定下来(如下图所示)。因此，一旦算法停止，最终的参数值是好的，但不是最优的。当代价函数非常不规则时，这实际上可以帮助算法跳出局部极小值，因此随机梯度下降比批量梯度下降有更好的机会找到全局最小值；
  >
  > 因此，从局部最优中逃离随机性是好的，但也意味着算法永远不能满足于最小值。解决这一困境的一个办法是逐渐降低学习速度。步骤一开始很大(这有助于快速进行并避免局部极小值)，然后变得越来越小，从而允许算法稳定在全局极小值。这一过程类似于**模拟退火**，模拟退火是一种算法，灵感来自于冶金中熔融金属缓慢冷却的退火过程。确定每次迭代的学习速度的函数称为 **learning schedule**。如果学习速度降低得太快，您可能会陷入局部最小值，甚至会冻结到最小值的一半。如果学习速度降低得太慢，你可能会在很长一段时间内跳过最小值，如果过早停止训练，最终会得到次优解。

  ![Stochastic Gradient Descent](C:\Users\FengZhang\AppData\Roaming\Typora\typora-user-images\1557997667890.png)

  ​																					**随机梯度下降法**

- 小批量梯度下降法 Mini-batch Gradient Descent

- 学习Mini-Batch与SGD

- 学习Batch与Mini-Batch，SGD梯度下降的区别

- 如何根据样本大小选择哪个梯度下降(批量梯度下降，Mini-Batch）？

- 写出SGD和Mini-Batch的代码

  `实现一个简单的随机梯度下降`

  ```python
  # Stochastic Gradient Descent
  
  n_epochs = 50
  t0, t1 = 5, 50 # learning schedule 超参数
  
  def learning_schedule(t):
      return t0 / (t + t1)
  
  theta = np.random.randn(2,1) # 随机初始化
  
  for epoch in range(n_epochs):
      for i in range(m):
          random_index = np.random.randint(m)
          xi = X_b[random_index:random_index+1]
          yi = y[random_index:random_index+1]
          gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
          eta = learning_schedule(epoch * m + i)
          theta = theta - eta * gradients
  ```

  `利用Scikit-Learn`

  ```python
  # Stochastic Gradient Descent with Scikit-Learn
  
  from sklearn.linear_model import SGDRegressor
  sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
  sgd_reg.fit(X, y.ravel())
  ```

  ![Gradient Descent 路径](C:\Users\FengZhang\AppData\Roaming\Typora\typora-user-images\1557997519224.png)

- 机器学习系统的类型
  - 监督 or 非监督学习
  - 批量学习 & 在线学习
  - 基于实例学习 vs. 基于模型学习

### 学习交叉验证

> **交叉验证（Cross Validation）是我们用来评估所选超参数组合的技术。 我们使用K折交叉验证，而不是将训练集分成单独的训练集和验证集，这会减少我们可以使用的训练数据量。交叉验证涉及到将训练数据分成K个folds，然后经历一个迭代过程，在这个迭代过程中，我们首先训练前K-1个folds的数据，然后在第K个fold的数据上进行评估表现。我们重复这个过程K次，在K-fold交叉验证结束时，我们将每个K次迭代的平均误差作为最终的性能度量。**

- 区分训练 & 验证
  - 超参调整 & 模型选择
  - 数据不匹配
- 使用交叉验证来选择模型 & 调整超参.
- 使用交叉验证度量精度 (性能度量)

### 学习归一化 

- 对比记忆 之 归一化 (Normalization) vs. 标准化 (Standardization)
  - 基于参数或者距离的模型都需要进行归一化处理，通过L1-norm或L2-norm将数值映射到[0, 1]区间。归一化化就是要把你需要处理的数据经过处理后限制在你需要的一定范围内；
    - 归一化后加快了梯度下降求最优解的速度，在梯度下降进行求解时能较快的收敛。如果不做归一化,梯度下降过程容易出现锯齿状,很难收敛甚至不能收敛；
    - 把有量纲表达式变为无量纲表达式, 有可能提高精度。一些分类器需要计算样本之间的距离(如欧氏距离),比如kNN,k-Means。如果一个特征值域范围非常大,那么距离计算就主要取决于这个特征,从而与实际情况背道而驰；
- 标准化（Standardization）和归一化（Normalization）的区别与联系
  - 标准化是依据特征矩阵的列处理数据，通过求其z-score的方法，转化为标准正态分布，和整体样本分布相关，每个样本点都能对标准化产生影响；
  - 归一化是依据特征矩阵的行处理数据，其目的在于样本向量在点乘运算或其他核函数计算相似性时，使其有统一的标准；
  - 标准化与归一化都能消除由于量纲不同引起的误差，都是一种线性变换，都是对向量按照比例压缩再进行平移。

### 学习模型评估

- **模型评估（Model Evaluation）**
- **模型评估主要分为离散评估和在线评估（针对分类、回归、排序、序列预测等问题）**
- **评估指标的局限性**

> **精准率的局限性：当不同类别的样本比例不均衡时，占比大的类别往往将影响其精准率**

> **精准率与召回率的权衡：**

> **精准率指分类正确的正样本个数占分类器判定为正样本的样本个数比例**

> **召回率指分类正确的正样本个数占真正的正样本个数的比例**

> **为了评估一个排序模型的好坏，不仅要看模型在不同的 Top N下的 Precision N 和 Recall N 同时还要绘制出模型的 P-R 曲线，通过 P-R 曲线的整体表现，对模型进行全面的评估**

> **ROC曲线（二值分类问题）：ROC曲线的横坐标为 假阳性率 FPR 纵坐标为真阳性率 TPR 其中 FPR = FP/N TPR = TP/P 其中，P为真实正样本数量 N为真实负样本数量 TP 是P个样本中被分类器预测为正样本的个数，FP 是N个负样本中被分类器预测为正样本的个数**

> **AUC的计算：AUC 指 ROC 曲线下的面积大小，其值能够量化的反映基于ROC曲线衡量出的模型性能**

> **ROC曲线与P-R曲线的特点：相比 P-R 曲线，当正负样本的分布发生变化时，ROC曲线的形状能够基本保持不变，而 P-R 曲线的形状一般会发生剧烈变化；如果选择不同的测试集， ROC 曲线能更加稳定的反映模型本身的好坏， 而 P-R 曲线能够直观的反映在特定数据集上的表现性能**

### 学习回归模型评价指标

- 常见的回归模型评价指标

  - **平均绝对误差**（Mean Absolute Error，MAE）：
    $$
    M A E=\frac{1}{N} \sum_{i=1}^{N}\left|y_{i}-\hat{y}_{i}\right|
    $$
    其中，N为样本个数，yi为第i个样本的真实值，y_head为第i个样本的预测值；

  - **均方误差**（Mean Squared Error，MSE）：
    $$
    M S E=\frac{1}{N} \sum_{i=1}^{N}\left(y_{i}-\hat{y}_{i}\right)^{2}
    $$
    

  - **平均绝对百分误差**（Mean Absolute Percentage Error，MAPE）：
    $$
    M A P E=\frac{100}{N} \sum_{i=1}^{N}\left|\frac{y_{i}-\hat{y}_{i}}{y_{i}}\right|, y_{i}=0
    $$
    MAPE通过计算绝对误差百分比来表示预测效果，其取值越小越好.

  - **均方根误差**（Root Mean Squared Error）：
    $$
    R M S E=\sqrt{\frac{1}{N} \sum_{i=1}^{N}\left(y_{i}-\hat{y}_{i}\right)^{2}}
    $$
    RMSE代表的是预测值和真实值差值的样本标准差.

  - **均方根对数误差**（Root Mean Squared Logarithmic Error，RMSLE）：

  $$
  R M S L E=\sqrt{\frac{1}{N} \sum_{i=1}^{N}\left(\log \left(y_{i}+1\right)-\log \left(\hat{y}_{i}+1\right)\right)^{2}}
  $$

  RMSLE对预测值偏小的样本惩罚比预测值偏大的样本惩罚更大.

- 选择 & 训练模型
  - 在训练集训练 & 评估
  - 使用交叉验证更好的评估

- 调整模型
  - 网格搜索
  
    > **网格搜索，是在所有候选的参数祖安泽中，通过循环遍历，尝试每种可能性，表现最好的参数就是最终的结果！**
  
    > **模型参数（Model Parameters）,是模型在训练过程中学习的内容**

    > **模型超参数（Model Hyperparameters），被认为是机器学习算法的最好设置，该算法是由数据科学家在训练之前设置的**
  
    > **通过改变模型中欠拟合和过拟合的平衡来控制影响模型性能的超参数。当我们的模型不够复杂（它没有足够的自由度）来学习从特征到目标的映射时，就是欠拟合（Underfitting）。一个欠拟合的模型有很高的偏置（bias），我们可以改变我们的模型使其更复杂来纠正**
  
    > **过拟合（Overfitting）是当我们的模型基本上拟合了所有训练数据点的时候。过拟合模型具有很高的方差（variance），我们可以通过正则化（regularization）来限制模型的复杂性来纠正。欠拟合和过拟合模型都不能很好地适用于测试数据（testdata）。**
  
  - 随机搜索
  
    > **随机搜索（Random Search）是指我们用来选择超参数的技术。我们定义一个网格，然后随机抽样不同的组合，而不是网格搜索（grid search），我们会彻底地尝试每一个组合。（令人惊讶的是，随机搜索的结果几乎和网格搜索一样，但大大缩短了运行时间。）**
  
  - 集成方法
  
    - Bagging & Pasting
    - 随机森林
    - Boosting
      - AdaBoost
      - Gradient Boosting
    - Stacking
  
  - 分析最好的模型 & 误差
  
  - 在测试集上评估系统
  
- 性能度量
  - 使用交叉验证度量精度 (已解析)
  - 混淆矩阵 (Matrix Confusion)
  - 查准率 (Precision) & 召回率 (Recall)
  - AUC 面积 & ROC 曲线

- 训练模型
  - 线性回归
    - 标准方程
    - 计算复杂度
    
  - 梯度下降 (已解析)
    - 批量梯度下降 (Bitch-Gradient-Descent, BGD)
    - 随机梯度下降 (Stochastic-Gradient-Descent, SGD)
    - 小批量梯度下降 (Mini-Batch-Gradient-Descent, MBGD)
    
  - 多元线性回归
  
  - 学习画布
  
  - 正规化 or 正则化线性模型
    
    提到正则化，最好的切入点就是“过拟合”，简单地理解过拟合就是模型过分学习并拟合数据导致模型泛化性能较差。通过正则化的方法，可以尽量避免过拟合的发生；
    
    - 正则化的概念
      - What
        正则化-Regularization（也称为惩罚项或范数）就是通过对模型的参数在“数量”和“大小”方面做相应的调整，从而降低模型的复杂度，以达到可以避免过拟合的效果。
    
    - 如何理解正则化
    
      - How
    
        如果我们的目标仅仅是最小化损失函数（即经验风险最小化），那么模型的复杂度势必会影响到模型的整体性能；引入正则化（即结构风险最小化）可以理解为衡量模型的复杂度，同时结合经验风险最小化，进一步训练优化算法。
    
    - 正则化的作用
      - Why
        正则化可以限制模型的复杂度，从而尽量避免过拟合的发生；模型之所以出现过拟合的主要原因是学习到了过多噪声，即模型过于复杂（也可以通过简化模型或增加数据集等方法尽量避免过拟合的发生）。
    - 正则化的常见类型
      - L1正则化
        可以通过稀疏化（减少参数“数量”）来降低模型复杂度的，即可以将参数值减小到0；
      - L2正则化
        可以通过减少参数值“大小”来降低模型的复杂度，即只能将参数值不断减小，但永远不会减小为0，只能尽量接近于0。
    
    - 岭回归
    - Lasso 回归
    - 弹性网络
    - 早停法则
    
  - 逻辑回归
    - 概率估计
    - 训练 & 代价函数
    - 决策边界
    - Softmax 回归