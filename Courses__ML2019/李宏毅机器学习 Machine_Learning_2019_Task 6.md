# 李宏毅机器学习 Machine_Learning_2019_Task 6

## 学习要求

- 公式手动推导
- 不掉包，手动实现算法
  - 独立手动创建数据，实现分类任务
  - 学习算法内容，对数据进行归一化操作
- 主要是学习LR算法的核心代码

## 学习内容

- 李航机器学习
- 机器学习实战
- 负责人笔记

### 方案 1 (参考负责人笔记)

```python
import numpy as np
import matplotlib.pyplot as plt

# 封装 LR 为一个类
class Logistic_Regression(object):

    def __init__(self, learning_rate=0.1, max_iter=100, seed=None):
        self.seed = seed
        self.lr = learning_rate
        self.max_iter = max_iter

    def fit(self, x, y):
        np.random.seed(self.seed)
        self.w = np.random.normal(loc=0.0, scale=1.0, size=x.shape[1])
        self.b = np.random.normal(loc=0.0, scale=1.0)
        self.x = x
        self.y = y
        for i in range(self.max_iter):
            self._update_step()

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def _f(self, x, w, b):
        z = x.dot(w) + b
        return self._sigmoid(z)

    def predict_proba(self, x=None):
        if x is None:
            x = self.x
        y_pred = self._f(x, self.w, self.b)
        return y_pred

    def predict(self, x=None):
        if x is None:
            x = self.x
        y_pred_proba = self._f(x, self.w, self.b)
        y_pred = np.array([0 if y_pred_proba[i] < 0.5 else 1 for i in range(len(y_pred_proba))])
        return y_pred

    # 定义 精度
    def score(self, y_true=None, y_pred=None):
        if y_true is None or y_pred is None:
            y_true = self.y
            y_pred = self.predict()
        acc = np.mean([1 if y_true[i] == y_pred[i] else 0 for i in range(len(y_true))])
        return acc

    # 定义损失函数
    def loss(self, y_true=None, y_pred_proba=None):
        if y_true is None or y_pred_proba is None:
            y_true = self.y
            y_pred_proba = self.predict_proba()
        return np.mean(-1.0 * (y_true * np.log(y_pred_proba) + (1.0 - y_true) * np.log(1.0 - y_pred_proba)))

    # 计算梯度
    def _calc_gradient(self):
        y_pred = self.predict()
        d_w = (y_pred - self.y).dot(self.x) / len(self.y)
        d_b = np.mean(y_pred - self.y)
        return d_w, d_b

    # 更新模型参数
    def _update_step(self):
        d_w, d_b = self._calc_gradient()
        self.w = self.w - self.lr * d_w
        self.b = self.b - self.lr * d_b
        return self.w, self.b

# 随机生成训练数据
def generate_data(seed):
    np.random.seed(seed)
    data_size_1 = 400
    x1_1 = np.random.normal(loc=5.0, scale=1.0, size=data_size_1)
    x2_1 = np.random.normal(loc=4.0, scale=1.0, size=data_size_1)
    y_1 = [0 for _ in range(data_size_1)]
    data_size_2 = 400
    x1_2 = np.random.normal(loc=10.0, scale=2.0, size=data_size_2)
    x2_2 = np.random.normal(loc=8.0, scale=2.0, size=data_size_2)
    y_2 = [1 for _ in range(data_size_2)]
    x1 = np.concatenate((x1_1, x1_2), axis=0)
    x2 = np.concatenate((x2_1, x2_2), axis=0)

    x = np.hstack((x1.reshape(-1, 1), x2.reshape(-1, 1)))
    y = np.concatenate((y_1, y_2), axis=0)
    data_size_all = data_size_1 + data_size_2
    shuffled_index = np.random.permutation(data_size_all)
    x = x[shuffled_index]
    y = y[shuffled_index]
    return x, y

# 划分训练集和测试集
def train_test_split(x, y):
    split_index = int(len(y) * 0.7)
    x_train = x[:split_index]
    y_train = y[:split_index]
    x_test = x[split_index:]
    y_test = y[split_index:]
    return x_train, y_train, x_test, y_test

x, y = generate_data(seed=272)
x_train, y_train, x_test, y_test = train_test_split(x, y)

# 数据归一化
x_train = (x_train - np.min(x_train, axis=0)) / \
          (np.max(x_train, axis=0) - np.min(x_train, axis=0))
x_test = (x_test - np.min(x_test, axis=0)) / \
         (np.max(x_test, axis=0) - np.min(x_test, axis=0))

# 创建分类器
clf = Logistic_Regression(learning_rate=0.1, max_iter=500, seed=272)
clf.fit(x_train, y_train)

# 结果可视化
split_boundary_func = lambda x: (-clf.b - clf.w[0] * x) / clf.w[1]
xx = np.arange(0.1, 0.6, 0.1)
cValue = ['g', 'b']
plt.scatter(x_train[:, 0], x_train[:, 1], c=[cValue[i] for i in y_train], marker='o')
plt.plot(xx, split_boundary_func(xx), c='red')
plt.show()

# 结果预测、输出精度和损失
y_test_pred = clf.predict(x_test)
y_test_pred_proba = clf.predict_proba(x_test)
print(clf.score(y_test, y_test_pred))
print(clf.loss(y_test, y_test_pred_proba))
```

### 方案 2

```python
'''
独立手动创建数据，实现分类任务
'''
import numpy as np
import matplotlib.pyplot as plt

def generate_data(seed):
    np.random.seed(seed)
    data_size_1 = 400 # 类别 1
    x1_1 = np.random.normal(loc=5.0, scale=1.0, size=data_size_1) # 特征 1
    # print(x1_1)
    x1_2 = np.random.normal(loc=4.0, scale=1.0, size=data_size_1)  # 特征 2
    y_1 = [0 for i in range(data_size_1)]
    # print(y_1)

    data_size_2= 400  # 类别 1
    x2_1 = np.random.normal(loc=10.0, scale=2.0, size=data_size_2)  # 特征 1
    x2_2 = np.random.normal(loc=8.0, scale=2.0, size=data_size_2)  # 特征 2
    y_2 = [1 for i in range(data_size_2)]

    x1 = np.concatenate((x1_1, x1_2), axis=0)
    print(x1.shape) # (800,)
    x2 = np.concatenate((x2_1, x2_2), axis=0)
    print(x2.shape) # (800,)

    # 合成新的数据集 从水平方向拼接
    x = np.hstack((x1.reshape(-1, 1), x2.reshape(-1, 1)))
    # x = normalize(x)
    print(x.shape)
    # 通过指定axis进行选择拼接方向
    y = np.concatenate((y_1, y_2), axis=0)
    print(y.shape)

    # 总数据
    data_size_all = data_size_1 + data_size_2

    # 随机打乱数据
    shuf_data = np.random.permutation(data_size_all)
    x = x[shuf_data]
    y = y[shuf_data]
    print(type(x)) # <class 'numpy.ndarray'>
    print(type(y)) # <class 'numpy.ndarray'>
    
    return  x, y

def accuracy(y, y_pred):
    y = y.reshape(y.shape[0], -1)
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    return np.sum(y == y_pred)/len(y)

def train_test_split(x, y):
    split_index = int(len(y)*0.7)
    x_train = x[:split_index]
    y_train = y[:split_index]
    x_test = x[split_index:]
    y_test = y[split_index:]
    return x_train, y_train, x_test, y_test

class LogisticRegression():
    """
    手写逻辑回归算法
    learning_rate 学习率
    Sigmoid 激活函数
    """
    def __init__(self, learning_rate=.1):
        self.w = None
        self.learning_rate = learning_rate
        self.sigmoid = Sigmoid()

    def fit(self, X, y, n_iterations=4000):
        # 在第一列添加偏置列，全部初始化为 1
        X = np.insert(X, 0, 1, axis=1)
        X = X.reshape(X.shape[0], -1)
        y = y.reshape(y.shape[0], -1)

        n_samples, n_features = np.shape(X)

        # 参数初始化 [-1/n_features, 1/n_features]
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, (n_features, 1))

        for i in range(n_iterations):
            # 通过初始化的参数 w 计算预测值
            y_pred = self.sigmoid.function(X.dot(self.w))
            # 梯度下降更新参数w.
            self.w -= self.learning_rate * \
                      X.T.dot(-(y - y_pred) * \
                        self.sigmoid.function(X.dot(self.w)) * \
                            (1 - self.sigmoid.function(X.dot(self.w))))


    def predict(self, X):
        # 训练模型时添加偏置，预测的时候也需要添加偏置
        X = X.reshape(X.shape[0], -1)
        X = np.insert(X, 0, 1, axis=1)
        # 预测
        y_pred = np.round(self.sigmoid.function(X.dot(self.w))).astype(int)
        return y_pred

x, y = generate_data(seed=20190601)
x_train, y_train, x_test, y_test = train_test_split(x, y)

# 数据归一化操作
x_train = (x_train - np.min(x_train, axis=0)) / \
          (np.max(x_train, axis=0) - np.min(x_train, axis=0))
x_test = (x_test - np.min(x_test, axis=0)) / \
         (np.max(x_test, axis=0) - np.min(x_test, axis=0))

clf = LogisticRegression()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
accu = accuracy(y_test, y_pred)
print ("Accuracy:", accu)

plt.figure(figsize=(12, 8))
plt.scatter(x[y==0][:,0], x[y==0][:,1])
plt.scatter(x[y==1][:,0], x[y==1][:,1])
plt.show()
```

### 方案 3 (利用 Scikit-Learn 的数据集)

```python
from __future__ import print_function, division
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
import math

'''
独立手动创建数据，实现分类任务
'''

def shuffle_data(X, y, seed=None):
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]

# 正规化数据集 X
def normalize(X, axis=-1, p=2):
    lp_norm = np.atleast_1d(np.linalg.norm(X, p, axis))
    lp_norm[lp_norm == 0] = 1
    return X / np.expand_dims(lp_norm, axis)

# 标准化数据集 X
def standardize(X):
    X_std = np.zeros(X.shape)
    mean = X.mean(axis=0)
    std = X.std(axis=0)

    # 做除法运算时请永远记住分母不能等于0的情形
    # X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    for col in range(np.shape(X)[1]):
        if std[col]:
            X_std[:, col] = (X_std[:, col] - mean[col]) / std[col]
    return X_std

# 划分数据集为训练集和测试集
def train_test_split(X, y, test_size=0.2, shuffle=True, seed=None):
    if shuffle:
        X, y = shuffle_data(X, y, seed)
    n_train_samples = int(X.shape[0] * (1-test_size))
    x_train, x_test = X[:n_train_samples], X[n_train_samples:]
    y_train, y_test = y[:n_train_samples], y[n_train_samples:]
    return x_train, x_test, y_train, y_test

# 将一个向量转换成对角阵，其中对角阵上的元素就是向量中元素
def vec2diagonal(vec):
    vec_length = len(vec)
    diagonal = np.zeros((vec_length, vec_length))
    for i in range(vec_length):
        diagonal[i][i] = vec[i]
    return diagonal

def accuracy(y, y_pred):
    y = y.reshape(y.shape[0], -1)
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    return np.sum(y == y_pred)/len(y)

class Sigmoid:
    def function(self, x):
        return 1/(1 + np.exp(-x))

    def derivative(self, x):
        return self.function(x) * (1 - self.function(x))

class LogisticRegression():
    """
    手写逻辑回归算法
    learning_rate 学习率
    Sigmoid 激活函数
    """
    def __init__(self, learning_rate=.1):
        self.w = None
        self.learning_rate = learning_rate
        self.sigmoid = Sigmoid()

    def fit(self, X, y, n_iterations=4000):
        # 在第一列添加偏置列，全部初始化为 1
        X = np.insert(X, 0, 1, axis=1)
        X = X.reshape(X.shape[0], -1)
        y = y.reshape(y.shape[0], -1)

        n_samples, n_features = np.shape(X)

        # 参数初始化 [-1/n_features, 1/n_features]
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, (n_features, 1))

        for i in range(n_iterations):
            # 通过初始化的参数 w 计算预测值
            y_pred = self.sigmoid.function(X.dot(self.w))
            # 梯度下降更新参数w.
            self.w -= self.learning_rate * \
                      X.T.dot(-(y - y_pred) * \
                        self.sigmoid.function(X.dot(self.w)) * \
                            (1 - self.sigmoid.function(X.dot(self.w))))

    def predict(self, X):
        # 训练模型时添加偏置，预测的时候也需要添加偏置
        X = X.reshape(X.shape[0], -1)
        X = np.insert(X, 0, 1, axis=1)
        # 预测
        y_pred = np.round(self.sigmoid.function(X.dot(self.w))).astype(int)
        return y_pred

def main():
    # 加载数据集
    data = datasets.load_iris()
    X = normalize(data.data[data.target != 0])
    y = data.target[data.target != 0]
    y[y == 1] = 0
    y[y == 2] = 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, seed=1)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accu = accuracy(y_test, y_pred)
    print ("Accuracy:", accu) # Accuracy: 0.9393939393939394

    plt.figure(figsize=(12, 8))
    plt.scatter(X[y==0][:,0], X[y==0][:,1])
    plt.scatter(X[y==1][:,0], X[y==1][:,1])
    plt.show()

if __name__ == "__main__":
    main()
```

![](D:\PyWorkplace\PyPro1\MachineLearning_In_Action\Logistic_Regression\Hands-on Classification\IMGs\Manual_Logistic_Regression.png)

### 方案 4

```python
'''
实现两个特征变量的LR算法的线性分类器（二分类问题）
'''
# 导入所需的包
import numpy as np
import matplotlib.pyplot as plt

# 生成模拟数据
z = np.linspace(-10, 10, 100)
sigm = 1./(1. + np.exp(-z))
result_plot = plt.plot(z, sigm)
result_visual = plt.ylim(0, 1)
plt.show()
# 加入随机噪声进一步得到模拟数据
np.random.seed(20190601)
m = 100   # 样本量
xlim = 4  # 数据采样范围
x0 = np.ones((m, 1))
x1 = np.random.rand(m, 1)*xlim*2 - xlim
x2 = np.random.rand(m, 1)*xlim*2 - xlim
y_ = 1./(1. + np.exp(-(x1 + x2))) + np.random.randn(m, 1)*0.2
y = np.round(y_)

# 作图可视化
_ = plt.scatter(x1, x2, c=y)
_ = plt.plot(x1, -x1, 'r')
_ = plt.ylim(-4.5, 3.5)
plt.show()

# 假设函数
def h_theta(X, theta):
    return 1. / (1. + np.exp(- np.dot(X, theta)))

# 损失函数
def loss_func(X, theta, y):
    y1 = np.log(h_theta(X, theta))
    y0 = np.log(1. - h_theta(X, theta))
    return -1. / m * (np.dot(y.T, y1) + np.dot((1. - y.T), y0))

# 梯度函数
def grad_func(X, theta, y):
    return 1. / m * np.dot(X.T, h_theta(X, theta) - y)

# 生成随机种子
np.random.seed(20190601)
# 设置学习率和收敛阈值
alpha = 0.1
stop = 1e-6

i = 1
index = 1
c = np.array([0.8, 0.8, 0.8])  # 设置颜色，颜色逐渐加深

theta = np.random.randn(3, 1)
X = np.hstack((x0, x1, x2))
grad = grad_func(X, theta, y)
while not np.all(abs(grad) <= stop):
    theta = theta - alpha * grad
    grad = grad_func(X, theta, y)

    # 作出学习过程
    i = i + 1
    if i % index == 0:
        yline = -theta[0] / theta[2] - theta[1] / theta[2] * x1
        _ = plt.plot(x1, yline, color=c)
        c = c - 0.1
        index = index * 4

res_scat = plt.scatter(x1, x2, c=y)
res_plt = plt.plot(x1, -x1, 'r')
res_visual = plt.ylim(-4.5, 3.5)
plt.show()
# 测试数据
np.random.seed(2019060134) #修改随机种子
test_x0 = np.ones((m, 1))
test_x1 = np.random.rand(m, 1)*xlim*2 - xlim
test_x2 = np.random.rand(m, 1)*xlim*2 - xlim
test_ty = 1./(1. + np.exp(-(test_x1 + test_x2))) + np.random.randn(m, 1)*0.2
test_y = np.round(test_ty)

test_X = np.hstack((test_x0, test_x1, test_x2))
y_ = h_theta(test_X, theta)
pre_y = np.round(y_)

acc = sum(int(a == b) for a, b in zip(pre_y, test_y))/m
print(acc) # 0.91
```

![](D:\PyWorkplace\PyPro1\MachineLearning_In_Action\Logistic_Regression\Hands-on Classification\IMGs\Sigmoid.png)

![](D:\PyWorkplace\PyPro1\MachineLearning_In_Action\Logistic_Regression\Hands-on Classification\IMGs\Scatter.png)

![](D:\PyWorkplace\PyPro1\MachineLearning_In_Action\Logistic_Regression\Hands-on Classification\IMGs\Scatter_Fit.png)

## Next Step Plan

### 熵

- 证明
- 计算信息熵(Machine Learning in Action P36)
- 学习联合概率和边缘概率
  - 推导条件熵公式
- 学习相对熵以及互信息
  - 理解交叉熵为什么能做为 Loss Function

### 决策树

- 李航统计学习 P55-P58
  - 总结决策树模型结构
  - 理解决策树递归思想
  - 学习信息增益以及信息增益率
  - 学习ID3、C4.5算法的优缺点
  - 理解C4.5在ID3上有何提升
  - 学习C4.5在连续值上的处理
  - 学习决策树的生成过程
- Machine Learning in Action
  - 手写划分数据集代码
  - 手动实现选择最好的数据集划分方式
  - 手动实现创建树的函数
  - 根据提供的数据创建树的图形

### CART分类回归树

- 李航统计学习 P65-67
  - 学习预剪枝
  - 学习后剪枝
  - 学习基尼系数
  - 学习CART的生成(回归树模型)

### 面试之灵魂拷问

> 算法十问

