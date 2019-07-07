# 李宏毅机器学习 Machine_Learning_2019_Task 7

## 学习导图

![](C:\Users\FengZhang\Desktop\ML2019\决策树 Decision Tree.png)

## 熵

## 决策树的构建算法

## 决策树的生成

## 决策树剪枝

## 灵魂十问

## 面试真题

### 熵

- 证明

  - 预备知识

    Jessen 不等式：
    $$
    E f(x) \geq f(E x)
    $$
    其中，f(x)为凸函数，Ex为期望值.

    对于离散型随机变量，可以将以上形式转换为：
    $$
    \sum_{i} p_{i} f\left(x_{i}\right) \geq f\left(\sum_{i} p_{i} x_{i}\right)
    $$
    其中，
    $$
    \sum_{i} p_{i} = 1,\ p_i \ge 0. \\
    \ p_i为概率分布.
    $$

  - 均匀分布熵最大
    $$
    0 \leqslant H(p) \leqslant \log n
    $$
    假设离散型随机变量X的分布为：X1 X2 ...... Xn，对应的概率分布为：p1 p2 ...... pn，且满足
    $$
    \sum_{i=1}^{n}p_{i}=1
    $$
    对于单一变量Xi，其对应的熵为-log(pi)，所有概率分布的熵为
    $$
    \sum_{i=1}^{n} p_{i} \cdot \log \frac{1}{p_{i}}=-\sum_{i=1}^{n} p_{i} \cdot \log p_{i}\\
    其含义就是期望
    $$
    由以上的 Jessen 不等式可知，
    $$
    p_{1} \log p_{1}+p_{2} \log p_{2}+\cdots+p_{n} \log p_{n} \ge \frac{p_{1}+p_{2}+\dots+p_{n}}{n} \log \frac{P_{1}+p_{2}+\ldots+n_{n}}{n}
    $$
    当且仅当，p1 = p2 = ...... = pn = 1/n 时，等号成立，即
    $$
    \sum_{i=1}^{n} p_{i} \cdot \log p_{i} \geqslant \log \frac{1}{n}
    $$
    通过变形，等价于
    $$
    0 \leqslant H(p)-\sum_{i=1}^{n} p_{i} \cdot \log p_{i} \leqslant -\log \frac{1}{n}=logn
    $$
    换言之，当随机变量Xi的概率分布pi取均匀分布(pi为1/n)时，熵最大！

    综上所述，证明完毕！

- 计算信息熵(Machine Learning in Action P36)

```python
import pandas as pd
from math import log

dataset = pd.read_csv("watermelon_3a.csv", header=None)
dataset = dataset.values

# 计算香农熵 度量数据集无序程度
def calcShannonEnt(dataSet):
    # 计算数据集中实例的总数
    numEntries=len(dataSet)
    # 为所有可能分类创建字典 其键值是最后一列的数值
    labelCounts={}
    for fearVec in dataSet:
         currentLabel=fearVec[-1] # 取最后一列键值，记录当前类别出现次数
         if currentLabel not in labelCounts.keys():
             # 扩展字典并将当前键值加入字典，每个键值都记录了当前类别出现的次数
             labelCounts[currentLabel] = 0
         labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries #该类别的概率
        shannonEnt-=prob*log(prob,2) #计算香农熵
    return shannonEnt

calShanEnt = calcShannonEnt(dataset)
print(calShanEnt) # 1.2516291673878228
```

- 学习联合概率和边缘概率(前期概率生成模型已解析)

  - 推导条件熵公式

  条件熵定义为随机变量X给定条件下，随机变量Y的条件概率分布的熵对X的数学期望，可以用来衡量在已知随机变量X的条件下随机变量Y的不确定性
  $$
  H(Y | X)=\sum_{I=1}^{m} p_{i} H\left(Y | X=x_{i}\right)
  $$
  推导：
  $$
  由信息增益的定义可知\\
  \mathrm{H}(\mathrm{X}, \mathrm{Y})=\mathrm{H}(\mathrm{X})-\mathrm{H}(\mathrm{Y} | \mathrm{X})
  $$
  移项变形之后可得：
  $$
  \begin{aligned} & H(Y | X)=H(X, Y)-H(X) \\ &=-\sum_{x, y} p(x, y) \log p(x, y)+\sum_{x} p(x) \log p(x) ......(1)\\ &=-\sum_{x, y} p(x, y) \log p(x, y)+\sum_{x}\left(\sum_{y} p(x, y)\right) \log p(x)......(2)\\ &=-\sum_{x, y} p(x, y) \log p(x, y)+\sum_{x, y} p(x, y) \log p(x)......(3)\\ &=-\sum_{x, y} p(x, y) \log \frac{p(x, y)}{p(x)}......(4)\\ &=-\sum_{x, y} p(x, y) \log p(y | x)......(5)\end{aligned}
  $$
  其中，

  (1) ——> (2) 是根据边缘概率分布p(x)等于联合概率分布p(x, y)之和；

  (2) ——> (3) 做了稍微的变形；

  (3) ——> (4) 是根据对数的性质得到的；

  (4) ——> (5) 是根据条件概率的性质得到的；

- 学习相对熵以及互信息

  相对熵：相对熵也称之为KL散度，用来衡量两个概率分布之间的差异，假设p(x)，q(x)为随机变量X取值中的两个概率分布，则p对q的相对熵用公式化表示为
  $$
  D(p \| q)=\sum_{x} p(x) \log \frac{p(x)}{q(x)}=E_{p(x)} \log \frac{p(x)}{q(x)}
  $$
  该值越低，表示训练出来的概率Q越接近于样本集概率P，即越准确！

  互信息：两个随机变量X，Y的互信息 **I(X,Y)** 定义为X，Y的联合概率分布与各自独立概率分布乘积的相对熵，可理解为I(X,Y) = D(P(X,Y) || P(X)P(Y))，公式表示为
  $$
  I(X, Y)=\sum_{x, y} p(x, y) \log \frac{p(x, y)}{p(x) p(y)}
  $$
  具体推导：
  $$
  \begin{array}{l}{H(Y)-I(X, Y)} \\ {=-\sum_{y} p(y) \log p(y)-\sum_{x, y} p(x, y) \log \frac{p(x, y)}{p(x) p(y)}} \\ {=-\sum_{y}\left(\sum_{x} p(x, y)\right) \log p(y)-\sum_{x, y} p(x, y) \log \frac{p(x, y)}{p(x) p(y)}} \\ {=-\sum_{x y} p(x, y) \log p(y)-\sum_{x, y} p(x, y) \log \frac{p(x, y)}{p(x) p(y)}} \\ {=-\sum_{x y} p(x, y) \log \frac{p(x, y)}{p(x)}} \\ {=-\sum_{x y} p(x, y) \log p(y | x)} \\ {=H(Y | X)}\end{array}
  $$
  有上式推导过程可知 H(Y)-I(X,Y) = H(Y|X)，而通过条件熵可知 H(Y|X) = H(X,Y) - H(X)，有互信息定义可知 H(Y|X) = H(Y) - I(X,Y)，最终整理可得 **I(X,Y)= H(X) + H(Y) - H(X,Y)**.

  - 理解 LR 交叉熵为什么能做为 Loss Function
    $$
    LR-Cross Entropy Loss Function:\\L(\theta)=\sum_{i=1}^{m}\left[y^{i} \operatorname{logh}_{\theta}\left(x^{i}\right)+\left(1-y^{i}\right) \log \left(1-h_{\theta}\left(x^{i}\right)\right)\right](交叉熵越小越好)
    $$
    在逻辑回归中利用 MSE(均方误差) 得到的损失函数往往是非凸的，不易于优化，容易陷局部极值点，而利用交叉熵的损失函数往往是一个凸函数，自变量的取值范围为 [0, 1]，且凸函数利于梯度下降和方向传播，便于优化，因此在 LR 分类问题中一般采用交叉熵作为损失函数！

### 决策树

决策树是一种自上而下，对样本数据进行树形分类的过程，由结点和有向边组成。结点分为内部结点和叶子结点，其中每个内部结点表示一个特征(属性)，叶子结点表示类别；

- ID3： 最大信息增益
- C4.5： 最大信息增益比
- C5.0： 与C4.5相比，它使用的内存更少，构建的规则集也更小，同时也更准确
- CART： 最大基尼指数

- 李航统计学习 P55-P58

  - 总结决策树模型结构

    对于决策树一般从顶部根结点开始，所有样本集中在一起，经过根节点的划分，样本被分到不同的子结点中，再根据子结点的特征进一步划分，如此递归地对样本进行测试并分配，直至所有样本都被归到某一类别中；

    决策树的生成一般包括特征选择，树的构造和树的剪枝三个过程.

  - 理解决策树递归思想

    ```pseudocode
    Generate_Decision_Tree(D, attribute_list)
    1 create a node N;
    2 if tuples in D are all of the same class C, then
    3 	return N as a leaf node labeled with the class C;
    4 if attribute_list is empty, then
    5 	return N as a leaf node labeled with the majority class in D; // majority
    voting
    6 apply Attribute_selection_method(D, attribute_list) to find the highest
    information gain;
    7 label node N with test-attribute;
    8 for each value ai of test-attribute
    9 	Grow a branch from node N for test-attribute = ai;
    10 	Let si be the set of samples in D for which test-attribute = ai;
    11 	if si is empty then
    12 		attach a leaf labeled with the majority class in D to node N;
    13 	else attach the node returned by Generate_Decision_
    Tree(si, attribute_list) to node N;
    14 end for
    ```

  - 学习信息增益以及信息增益率

    **ID3**是采用**信息增益**作为评价标准，信息增益反映的是给定条件以后不确定性减少的程度，特征取值越多就意味着确定性更高，也就是条件熵越小，信息增益越大(实际应用中存在缺陷)；

    **C4.5**实际上是对ID3进行优化，通过引入**信息增益率**，一定程度上对取值比较多的特征进行惩罚(对信息增益作进一步**归一化**处理)，避免ID3出现过拟合的特性，提升决策树的泛化能力.

  - 学习ID3、C4.5算法的优缺点

    ID3算法，核心是在决策树的各级结点上，使用信息增益作为属性的选择准则，进而帮助生成每个结点采用的合适属性，ID3相当于用极大似然法进行概率模型的选择，但ID3只有树的生成，因此生成的树容易产生过拟合；ID3只适用于离散型特征；ID3对样本特征缺失值比较敏感；C4.5算法相对于ID3的改进是使用信息增益率来选择结点属性，C4.5既能处理离散属性，也能处理连续属性.

    - 从样本类型分析，**ID3**只能处理**离散型变量**，而**C4.5**和**CART**都可以处理**连续型变量**。C4.5处理连续型变量时，通过对数据**排序**之后找到**类别不同**的**分割线**作为切分点，根据切分点把**连续属性**转换为**布尔型**，从而将**连续型变量**转换多个取值区间的**离散型变量**。而对于CART，由于其构建时每次都会对特征进行**二值划分**，因此可以很好地适用于**连续性变量**；
    - 从应用方面分析，**ID3**和**C4.5**只能用于**分类任务**，而**CART**不仅可以用于**分类**，也可以应用于**回归任务**(回归树使用最小平方误差准则)；
    - 从实现细节分析：
      - ID3对样本特征缺失值比较敏感，而C4.5和CART可以对缺失值进行不同方式的处理；
      - ID3和C4.5可以在每个结点上产生出多叉分支，且每个特征在层级之间不会复用，而CART每个结点只会产生两个分支，因此最后会形成一颗二叉树，且每个特征可以被重复使用；
      - ID3和C4.5通过剪枝来权衡树的准确性与泛化能力，而CART直接利用全部数据发现所有可能的树结构进行对比.

  - 理解C4.5在ID3上有何提升(已解释)

  - 学习C4.5在连续值上的处理(已解释)

  - 学习决策树的生成过程
    构建决策树的核心问题是在每一步如何选择适当的属性对样本做划分。对于分类问题，从已知类标记的训练样本中学习并构造出决策树是一个自上而下，分而治之的过程；决策树的学习算法一般包含特征选择、决策树的生成和剪枝过程。其学习算法通常是**递归**地选择**最优特征**，用最优特征对数据集进行**划分**

    - 首先构建根节点，选择最优特征，根据不同的特征值划分不同子集；
    - 每个子集分别递归调用此方法，返回节点，返回的节点即为上一层的子节点；
    - 直到所有特征都被用完，或数据集只有一维特征为止。

- Machine Learning in Action

  - 手写划分数据集代码

    ```python
    # 按照给定特征划分数据集
    # param1,param2,param3:待划分的数据集、划分数据集的特征、需要返回的特征的值
    def splitDataSet(dataSet, axis, value):#待划分的数据集 数据集特征 需要返回的特征值
        # 创建一个新的列表对象
        retDataSet = []
        for featVec in dataSet:
            if featVec[axis] == value:
                 reducedFeatVec = featVec[:axis]  # chop out axis used for splitting
                 reducedFeatVec.extend(featVec[axis + 1:])#extend方法是将添加元素融入集合
                 retDataSet.append(reducedFeatVec)#append将添加元素作为一个元素加入
        return retDataSet
    ```

  - 手动实现选择最好的数据集划分方式

    ```python
    #选择最好的数据集划分
    def chooseBestFeatureToSplit(dataSet):
        numFeatures = len(dataSet[0])-1
        baseEntropy = calcShannonEnt(dataSet)
        bestInfoGain = 0.0; bestFeature = -1
        for i in range(numFeatures):
            featList = [example[i] for example in dataSet] #使用列表推导来创建新的列表
            uniqueVals = set(featList) #python的集合set数据类型保存，从列表中创建集合来获取列表中的唯一元素值
            newEntropy = 0.0
            for value in uniqueVals:#遍历当前特征中的所有唯一属性值
                subDataSet = splitDataSet(dataSet,i,value)#对每个特征划分数据集
                prob=len(subDataSet)/float(len(dataSet))
                newEntropy += prob*calcShannonEnt(subDataSet)
            infoGain = baseEntropy - newEntropy
            if(infoGain > bestInfoGain):
                bestInfoGain = infoGain
                bestFeature = i
            return bestFeature
    ```

  - 手动实现创建树的函数

    ```python
    # 创建决策树
    def createTree(dataSet,labels):
        classList = [example[-1] for example in dataSet]
        if classList.count(classList[0]) == len(classList):
            return classList[0]  # stop splitting when all of the classes are equal
        if len(dataSet[0]) == 1:  # 使用完了所有的特征
            return majorityCnt(classList) # 返回出现次数最多的特征
        # 创建树
        bestFeat = chooseBestFeatureToSplit(dataSet) # 将选取的最好特征放在bestFeat中
        bestFeatLabel = labels[bestFeat]   # 特征标签
        myTree = {bestFeatLabel:{}}      # 使用特征标签创建树
        del(labels[bestFeat])  # del用于list列表操作，删除一个或者连续几个元素
        featValues = [example[bestFeat] for example in dataSet]
        uniqueVals = set(featValues)
        for value in uniqueVals:
            subLabels = labels[:]  # copy all of labels, so trees don't mess up existing labels
            myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
        return myTree
    ```

  - 根据提供的数据创建树的图形

    ```python
    # 绘制属树形图 递归函数
    def plotTree(myTree, parentPt, nodeTxt):
        numLeafs = getNumLeafs(myTree)  # 计算树的宽度  totalW
        depth = getTreeDepth(myTree) # 计算树的高度 存储在totalD
        firstSides = list(myTree.keys()) # firstStr = myTree.keys()[0]
        firstStr = firstSides[0]  # 找到输入的第一个元素
        # 按照叶子结点个数划分x轴
        cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
        plotMidText(cntrPt, parentPt, nodeTxt) # 标注子结点属性值
        plotNode(firstStr, cntrPt, parentPt, decisionNode)
        secondDict = myTree[firstStr]
        # y方向上的摆放位置 自上而下绘制，因此递减y值
        plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD 
        for key in secondDict.keys():
            if type(secondDict[key]).__name__=='dict': # 判断是否为字典 不是则为叶子结点
                plotTree(secondDict[key],cntrPt,str(key)) # 递归继续向下找
            else:   # 为叶子结点
                plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW # x方向计算结点坐标
                # 绘制
                plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
                # 添加文本信息
                plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
        # 下次重新调用时恢复y
        plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
    ```

### CART分类回归树

- 李航统计学习 P65-67

  通过降低决策树的复杂度来避免过拟合的过程称为剪枝.

  - 学习预剪枝(存在欠拟合的风险)

    预剪枝，即在生成决策树的过程中提前停止树的增长，其核心思想是在树中结点进行扩展之前，先计算当前的划分是否能够带来模型泛化能力的提升，若不能，则不再继续生长子树；若存在不同类别的样本同时存于结点中，则按多数投票原则判断该结点所属类别。停止决策树生长的方法：

    1. 当树到达一定**深度**的时候，停止树的生长；
    2. 当到达当前结点的样本数量小于某个**阈值**的时候，停止树的生长；
    3. 计算每次分裂对测试集的准确度提升，当小于某个阈值的时候，不再继
       续扩展.

  - 学习后剪枝(开销较大)

    后剪枝，即在已生成的过拟合决策树上进行剪枝，得到简化后的剪枝决策树，其核心思想是让算法生成一棵完全生长的决策树，然后从最底层向上计算是否剪枝。剪枝过程将子树删除，用一个**叶子结点**替代，该结点的类别同样按照多数投票的原则进行判断。常见方法：

    1. REP；
    2. PEP；
    3. CCP.

  - 学习基尼指数
  
    基尼指数描述的是数据的纯度，与信息上含义类似
    $$
    \operatorname{Gini}(D)=1-\sum_{k=1}^{n}\left(\frac{\left|C_{k}\right|}{|D|}\right)^{2}
    $$
    CART在每一次迭代中选择基尼指数最小的特征及其对应的切分点进行分类，与ID3、C4.5不同的是，CART是一颗二叉树，采用二元切割法，每一步将数据按特征A的取值切成两份，分别进入左右子树。特征A的Gini指数定义为：
    $$
    \operatorname{Gini}(D | A)=\sum_{i=1}^{n} \frac{\left|D_{i}\right|}{|D|} \operatorname{Gini}\left(D_{i}\right)
    $$
  
  - 学习CART的生成(回归树模型)
  
    **最小二乘回归树生成算法**

### 面试之灵魂十问

> [0]: https://github.com/datawhalechina/Daily-interview/blob/master/machine-learning/DecisionTree.md	"算法十问"

### 参考文献

[^]: 机器学习实战
[^]: 统计学习方法
[^]: 百面机器学习



