# chapter7 Gradient Descent (Demo by AOE)

- 利用**帝国时代**的方式模拟**梯度下降**；
- 在地图上大多数位置我们是未知的，只有我们单位走过的地方是可知；
- 地图上的海拔可以看作损失函数**loss function**，我们的目的就是寻找海拔的最低点的值；
- 随机初始一个位置，朝向较低的方向移动，周而复始，直到**local minimal**(在不开天眼的情况下，你始终不会知晓所在位置是否为global minimal)。

