# 李宏毅机器学习 Machine_Learning_2019_Task 4

## 学习目标

![](C:\Users\FengZhang\Desktop\ML2019\Classification 分类 .png)

### 理解概率模型

- 从**基础概率**推导**贝叶斯公式**以及**朴素贝叶斯**公式

  - **贝叶斯推论**

    - 什么是**贝叶斯定理**(**Bayes Theorem**)

      - 官方解释：贝叶斯定理是关于事件A与事件B的条件概率和边缘概率的一项准则 or 定理

      - 意义解释：利用我们已有的知识(也称**先验知识** or **先验信念**)帮助计算相关事件的**概率**

      - 数学表示：
        $$
        P(A | B)=\frac{P(B | A) \times P(A)}{P(B)}
        $$
        其中，

        - P(A|B) 指在 B 发生的情况下 A 发生的可能性，即已知 B 发生后 A 的条件概率，也可以理解为先有 B 再有 A，由于源于 B 的取值而被称作 A 的**后验概率**
        - P(A) 指 A 的**先验概率**或**边缘概率**(先验可以理解为事件 A 的发生不考虑任何 B 方面的因素)
        - P(B) 指 B 的**先验概率**或**边缘概率**，也可以作为**标准化常量**
        - P(B|A) 指已知 A 发生后 B 的条件概率，即先有 A 再有 B，由于源于 A 的取值而被称作 B 的**后验概率**

        贝叶斯定理可以形象地描述为：**后验概率** = (**相似度*****先验概率**) / **标准化常量**，即**后验概率**与**相似度和先验概率的乘积成正比**，由于 **P(B|A) / P(B)** 也被称作**标准相似度**，因此贝叶斯定理也可表述为：后验概率 = **标准相似度** * **先验概率**

      - 小试牛刀x1 (贝叶斯定理示例)

        假定一副扑克牌里有 52 张牌，其中 26 张是红色的，26 张是黑色的。那么当牌是红色的时候，牌上数字为 4 的概率是多少？

        我们『将牌为数字 4 设为事件 A』，『将牌为红色设为事件 B』。此时我们需要计算的是概率 P(A|B) = P(4|Red)，利用贝叶斯定理可计算出该概率值：

        - P(B|A) = P(Red|4) = 1/2
        - P(A) = P(4) = 4/52 = 1/13
        - P(B) = P(red) = 1/2
        - P(4|Red)=P(Red|4)*P(4) / P(Red) = 1/13 『 **Bayes Theorem**: 』

  - **先验概率**(已解释)

  - **后验概率**(已解释)

  **Note: **

  - **先验知识本身并不是完全客观的，可能带有主观成分，甚至是完全的猜测。而这也会对最终的条件概率计算产生影响!!!**；
  - **贝叶斯定理**既可以解决**分类**问题，也可以解决**回归**问题；
  - 必须确保输入变量之间是**条件独立**的.

  - 贝叶斯推理

    - 定义

      推理 or 统计 是从数据中推导群体分布 or 概率分布的性质的过程。比如，最大似然估计，其可以通过一系列的观察数据点确定平均值的最大似然估计。因此，贝叶斯推理也是利用贝叶斯定理从数据中推导群体分布或概率分布的性质的过程.

    - 使用贝叶斯定理处理数据分布

      概率分布可分为离散型概率分布和连续性概率分布。对于离散型概率分布我们可以指定事件发生的可能性；而对于连续型概率分布，其可以为任何值，每个概率值对应一个先验信念，很自然的用函数的形式 f(x) 表示，以下分布也称为先验分布 (**Prior Distribution**)![Prior Distribution](file:///C:/Users/FengZhang/Desktop/Prior%20Distribution.png)

         																	  **Prior Distribution**

    - 贝叶斯定理的模型形式

      我们将用 **Θ** 取代事件 **A**，**Θ** 表示**参数**的**集合**。如果要估计**高斯分布**的**参数值**，则 **Θ** 代表了平均值 **μ** 和标准差 **σ**，用数学形式表示为 **Θ = {μ, σ}**

      我们用 **data** 或 **y={y1, y2, …, yn}** 取代事件 **B**，它代表了观察数据的集合
      $$
      P(\Theta | d a t a)=\frac{P(d a t a | \Theta) \times P(\Theta)}{P(d a t a)}
      $$
      同理，

      - P(Θ) 是先验分布，其代表了我们相信的参数值分布
      - 等式左边的 P(Θ|data) 称为后验分布，其代表利用观察数据计算了等式右边之后的参数值分布
      - P(data| Θ) 和似然度分布类似

      **Note: 对于 P(data) 的解释**

      - 我们只对**参数的分布**感兴趣，而 P(data) 对此并没有任何参考价值

      - P(data) 的真正重要性在于它是一个**归一化常数** or **标准化常量**，它确保了计算得到的后验分布的总和等于 1 

      - 在某些情况下，我们并不关心归一化，因此可以将贝叶斯定理写成这样的形式
        $$
        P(\Theta | d a t a) \propto P(d a t a | \Theta) \times P(\Theta)
        $$
        其中 **∝** 表示符号左边正比于符号右边的表达式

    - 小试牛刀x2 (贝叶斯推理示例 之 计算氢键键长，你无需知道 氢键是 神马东东-vv-)

      假设氢键是 3.2Å—4.0Å。该信息将构成问题的先验知识。就概率分布而言，将其形式化为均值

      μ = 3.6Å，标准差 σ = 0.2Å 的高斯分布(为何使用高斯分布，请参照前期Task)![PB](file:///C:/Users/FengZhang/Desktop/%E5%85%88%E9%AA%8C%E5%88%86%E5%B8%83.png)

       																					**氢键键长的先验分布**

      我们现在选取一些数据（由均值为 3Å 和标准差为 0.4Å 的高斯分布随机生成的 5 个数据点），代表了氢键的测量长度（下图中的黄色点）。我们可以从这些数据点中推导出似然度分布，即下图中黄色线表示的似然度分布。注意从这 5 个数据点得到的最大似然度估计小于 3Å（大约 2.8Å）![似然度分布](file:///C:/Users/FengZhang/Desktop/%E6%B0%A2%E9%94%AE%E9%95%BF%E5%BA%A6%E7%9A%84%E5%85%88%E9%AA%8C%E5%88%86%E5%B8%83%EF%BC%88%E8%93%9D%E7%BA%BF%EF%BC%89%EF%BC%8C%E5%92%8C%E7%94%B1%205%20%E4%B8%AA%E6%95%B0%E6%8D%AE%E7%82%B9%E5%AF%BC%E5%87%BA%E7%9A%84%E4%BC%BC%E7%84%B6%E5%BA%A6%E5%88%86%E5%B8%83%EF%BC%88%E9%BB%84%E7%BA%BF%EF%BC%89.png)

      ​                      **氢键长度的先验分布（蓝线），和由 5 个数据点导出的似然度分布（黄线）**

      现在我们有两个高斯分布。由于忽略了归一化常数，因此已经可以计算非归一化的后验分布了。高斯分布的定义如下
      $$
      P(x ; \mu, \sigma)=\frac{1}{\sigma \sqrt{2 \pi}} \exp \left(-\frac{(x-\mu)^{2}}{2 \sigma^{2}}\right)
      $$
      我们需要将上述的两个分布乘起来，然后得到下图的粉线所示的后验分布![后验概率](file:///C:/Users/FengZhang/Desktop/%E8%93%9D%E8%89%B2%E5%88%86%E5%B8%83%E5%92%8C%E9%BB%84%E8%89%B2%E5%88%86%E5%B8%83%E7%9A%84%E4%B9%98%E7%A7%AF%E5%BE%97%E5%88%B0%E7%B2%89%E8%89%B2%E7%9A%84%E5%90%8E%E9%AA%8C%E5%88%86%E5%B8%83.png)

        												**蓝色分布和黄色分布的乘积得到粉色的后验分布**

      现在我们得到了氢键键长的后验分布，可以从中推导出统计特征..

  - **朴素贝叶斯**

    - 定义

      **朴素贝叶斯**是基于**贝叶斯定理**与**特征条件独立**假设的分类方法，对于给定的训练数据集，首先基于**特征条件独立假设**学习输入/输出的**联合概率分布**；然后基于此模型，对给定的输入实例x，利用贝叶斯定理求出后验概率最大的输出y.

    - 基本方法与算法

      - 基本方法

        设输入空间
        $$
        \mathcal{X} \subseteq \mathbf{R}^{n}
        $$
        为 n 维向量的集合，输出空间为类标记集合
        $$
        y_{i} \in\left\{c_{1}, c_{2}, \cdots, c_{K}\right\}.
        $$
        输入为特征向量
        $$
        x \in \mathcal{X}
        $$
        输出为类标签
        $$
        y \in \mathcal{Y}
        $$

        $$
        X 是 定义在输入空间\mathcal{X}上的随机向量，\\
        Y 是定义在输出空间\mathcal{Y}上的随机变量.\\
        P(X,Y)是X和Y的联合概率分布，训练集是\\
        T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \cdots,\left(x_{N}, y_{N}\right)\right\} \\
        由P(X,Y)独立产生.
        $$

        **朴素贝叶斯方法是通过训练数据集学习联合概率分布P(X,Y).**

        具体来说，就是指学习以下的先验概率分布以及条件概率分布：

        - **先验概率分布**
          $$
          P\left(Y=c_{k}\right), \quad k=1,2, \cdots, K
          $$

        - **条件概率分布**(条件概率分布有指数级数量的参数，其估计实际上是不可取的！！！)
          $$
          P\left(X=x | Y=c_{k}\right)=P\left(X^{(1)}=x^{(1)}, \cdots, X^{(n)}=x^{(n)} | Y=c_{k}\right), \quad k=1,2, \cdots, K
          $$

        - **联合概率分布**由先验概率和条件概率分布得知.

      朴素贝叶斯之所以称为朴素，是因为朴素贝叶斯方法是对条件概率分布作了条件独立性假设，由于这是一个较强的假设，因此朴素贝叶斯因此而得名。具体来说，条件独立性假设为：
      $$
      \begin{aligned} P\left(X=x | Y=c_{k}\right) &=P\left(X^{(1)}=x^{(1)}, \cdots, X^{(n)}=x^{(n)} | Y=c_{k}\right) \\ &=\prod_{j=1}^{n} P\left(X^{(j)}=x^{(j)} | Y=c_{k}\right) \end{aligned}
      $$
      确定实例对应的类别：
      $$
      y=\arg \max _{a} P\left(Y=c_{k}\right) \prod_{j=1}^{n} P\left(X^{(j)}=x^{(j)} | Y=c_{k}\right)
      $$

      - 算法(Naive Bayes Algorithm)![Naive Bayes Algorithm](file:///C:/Users/FengZhang/Desktop/Naive%20Bayes%20Algorithm.png)

        ​																**朴素贝叶斯算法**

  - 高斯朴素贝叶斯
  - 多项式朴素贝叶斯
  - 贝叶斯信念网络(后续补充)
  - 贝叶斯网络(后续补充) 属于 图模型的范畴(最近GCN or GNN 图神经网络如如后春笋，实用价值很高)
  - 图模型(尽管模型清晰，但很难确定其依赖关系)
    - 马尔科夫随机域
    - 链图
  - 图神经网络(GNN)
    - 图卷积网络(GCN)
    - 图神经网络(GNN)

- **判别模型(Descriminative Model)** vs. **生成模型(Generative Model)**

  - **判别模型**
    - 判别模型是一种**对观测数据进行直接分类的模型**，常见的模型有**逻辑回归(LR)**和**感知机学习算法(SVM)**等。此模型**仅对数据进行分类**，并**不能具象化** or **量化数据本身的分布状态**，因此也无法根据分类**生成**可观测的**新数据**；
    - 从定义上来说，判别模型通过构建条件概率分布 p(y|x;θ) 预测 y，即在特征 x 出现的情况下标记 y 出现的概率。此处 p 可以是逻辑回归模型.
  - **生成模型**
    - 与判别模型不同，生成模型是**先了解数据本身分布情况**，并进一步根据输入 x，给出预测分类 y 的概率。该模型有着**研究数据分布形态**的概念，可以根据历史数据**生成**可观测的**新数据**；
    - **贝叶斯分类**就是一个典型的例子。在这个例子中，我们有一个**先验分类**，根据这个先验分类，我们可以使用贝叶斯原理**计算**每个分类的**概率**，然后取对应概率最高的类别。同时，我们还可以**根据特定的先验生成特征**。这就是一个**生成过程**.

### 学习逻辑回归 Logistic Regression 算法

- 线性回归

- 正则化(回顾前期Task)

- 岭回归与Lasso回归(回顾前期Task)

- 逻辑回归

- 学习**逻辑回归**与**线性回归**之间的**区别**以及 **Logistic Regression** **梯度下降** (结合对比**线性回归**的**梯度下降法**)

  - 线性回归是输入到输出的线性变换，其计算的是具体的值，拟合能力有一定限制性，而逻辑回归通过阈值判断的方法，引入了非线性因素，可以处理分类问题(二分类 or 多分类问题)；

  - 线性回归模型是分析一个变量与另一个变量 or 多个变量(指自变量)之间关系强度的方法，而逻辑回归模型只输出数据点在一个 or 另一个类别中的“**概率**”，而不是常规数值，同时逻辑回归中的因变量与自变量的对数概率具有线性关系；

  - 线性回归的目标值是(−∞, +∞)，而逻辑回归的输出一般为(0, 1)；

  - 逻辑“回归” 是 一种 “分类”模型，其本质是在线性回归的基础上，引入了一个逻辑函数(比如，Sigmoid函数，把y的值从线性回归的(−∞, +∞)限制到了(0, 1)的范围) or 激活函数，用来估计某事件发生的“**可能性**”(该可能性指的是特征值的加权求和)。换言之，逻辑回归解决的是分类问题，而不是回归问题(对于初学者，估计很难接受这一事实)；

  - ![Gradient Descent可视化](C:\Users\FengZhang\Desktop\WeChat Image_20190525004352.gif)

    ​																		    **Gradient Descent可视化**

  **RSS**指的是点与线之间差异的平方和，该值代表了点与线的距离大小，而梯度下降就是要找出**RSS**的最小值.

- 推导 **Sigmoid** 函数公式，理解逻辑回归分类器

  - ![Logistic分布](C:\Users\FengZhang\Desktop\11-15 Logistic Regression.jpg)

    ![Logistic分布](C:\Users\FengZhang\Desktop\11-15 Logistic Regression1.jpg)

    ​														**不同参数对Logistic分布的影响(维基百科)**

    可以看到μ影响的是中心对称点的位置，γ越小中心点附近增长的速度越快，而Sigmoid是一种非线性变换函数，即Logistic分布的**γ=1,μ=0**的特殊形式(具体推导请回顾Task1)

### 学习多样本的向量计算

- 基于实例的算法 or 基于记忆的学习
  - k近邻(kNN)
  - 学习向量量化
  - 局部加权学习
- 理解**向量化**的优点 (One-hot 编码)
- 比较**循环**与**向量化**之间的区别

### 学习 Softmax 激活函数

- **Softmax 原理**
- **Softmax 损失函数**
- **Softmax 梯度下降**

### 学以致用

- 手动**获取数据**，基于得到的数据实现 **Logistic Regression** 逻辑回归算法

  ```python
  '''
  @Date：Created on May 25, 2019
  @Manual Coding：Logistic Regression
  @Description：手动获取数据，基于得到的数据实现 Logistic Regression 逻辑回归算法
  @Author: 追风者
  '''
  from numpy import *
  
  def loadDataSet():
      dataMat = []; labelMat = []
      fr = open('testSet.txt')
      for line in fr.readlines():
          lineArr = line.strip().split()
          dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
          labelMat.append(int(lineArr[2]))
      return dataMat,labelMat
  
  def sigmoid(inX):
      return 1.0/(1+exp(-inX))
  
  def gradAscent(dataMatIn, classLabels):
  	# 转换为NumPy矩阵
      dataMatrix = mat(dataMatIn)
      labelMat = mat(classLabels).transpose()
      m,n = shape(dataMatrix)
      alpha = 0.001
      maxCycles = 500
      weights = ones((n,1))
      for k in range(maxCycles):  
      	# 矩阵相乘
          h = sigmoid(dataMatrix*weights)    
          # 向量减法
          error = (labelMat - h)             
          weights = weights + alpha * dataMatrix.transpose() * error
      return weights
  
  def plotBestFit(weights):
      import matplotlib.pyplot as plt
      dataMat,labelMat=loadDataSet()
      dataArr = array(dataMat)
      n = shape(dataArr)[0] 
      xcord1 = []; ycord1 = []
      xcord2 = []; ycord2 = []
      for i in range(n):
          if int(labelMat[i])== 1:
              xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
          else:
              xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
      fig = plt.figure()
      ax = fig.add_subplot(111)
      ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
      ax.scatter(xcord2, ycord2, s=30, c='green')
      x = arange(-3.0, 3.0, 0.1)
      y = (-weights[0]-weights[1]*x)/weights[2]
      ax.plot(x, y)
      plt.xlabel('X1'); plt.ylabel('X2');
      plt.show()
  
  def stocGradAscent0(dataMatrix, classLabels):
      m,n = shape(dataMatrix)
      alpha = 0.01
      # 初始化为1
      weights = ones(n) 
      for i in range(m):
          h = sigmoid(sum(dataMatrix[i]*weights))
          error = classLabels[i] - h
          weights = weights + alpha * error * dataMatrix[i]
      return weights
  
  def stocGradAscent1(dataMatrix, classLabels, numIter=150):
      m,n = shape(dataMatrix)
      weights = ones(n)
      for j in range(numIter):
          dataIndex = range(m)
          for i in range(m):
          	# apha随着迭代次数的增加而减少,但不能到达0
              alpha = 4/(1.0+j+i)+0.0001 
              randIndex = int(random.uniform(0,len(dataIndex)))
              h = sigmoid(sum(dataMatrix[randIndex]*weights))
              error = classLabels[randIndex] - h
              weights = weights + alpha * error * dataMatrix[randIndex]
              del(dataIndex[randIndex]) # 存在问题
      return weights
  
  def classifyVector(inX, weights):
      prob = sigmoid(sum(inX*weights))
      if prob > 0.5: return 1.0
      else: return 0.0
  
  def colicTest():
      frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
      trainingSet = []; trainingLabels = []
      for line in frTrain.readlines():
          currLine = line.strip().split('\t')
          lineArr =[]
          for i in range(21):
              lineArr.append(float(currLine[i]))
          trainingSet.append(lineArr)
          trainingLabels.append(float(currLine[21]))
      trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
      errorCount = 0; numTestVec = 0.0
      for line in frTest.readlines():
          numTestVec += 1.0
          currLine = line.strip().split('\t')
          lineArr =[]
          for i in range(21):
              lineArr.append(float(currLine[i]))
          if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
              errorCount += 1
      errorRate = (float(errorCount)/numTestVec)
      print("the error rate of this test is: %f" % errorRate)
      return errorRate
  
  def multiTest():
      numTests = 10; errorSum=0.0
      for k in range(numTests):
          errorSum += colicTest()
      print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))
  ```


### 数学基础

- **链式求导**法则
- 基础**概率**与**统计**知识
- **矩阵乘法**与**矩阵求导**
