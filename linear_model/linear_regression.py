from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression as LR

import numpy as np


class LinearRegression:
    """
    线性回归使用闭式解
    注意在推导的时候X是数据集中的所有特征向量的矩阵拼接最后一列为1，
    W是权重和bias拼接的矩阵
    所以 y = X * W
    """

    def __init__(self):
        self.w = None

    def fit(self, X, y):
        """
        使用最小二乘法训练模型
        """
        self.w = np.linalg.inv(X.T @ X) @ X.T @ y  # @表示矩阵乘法

    def predict(self, X):
        """
        使用训练好的模型预测
        """
        return X @ self.w


# 加载数据
# 根据患者的一些生理指标预测患者一年后的糖尿病进展程度
data = load_diabetes()

# 输入是患者当前的身体特征
X = data.data  # 行数为样本数，列数为特征数
# 输出事一年后的疾病发展评分
y = data.target

print(X.shape)
print(y.shape)


# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_b = np.c_[
    X_train,
    np.ones((len(X_train), 1))
]

X_test_b = np.c_[
    X_test,
    np.ones((len(X_test), 1))
]

# 自己的model，使用闭式解
model = LinearRegression()
model.fit(X_train_b, y_train.reshape(-1, 1))  # 这里转置是因为y是一个横向的一维数组
# sklearn的model
sk_model = LR()
sk_model.fit(X_test_b, y_test.reshape(-1, 1))

y_pred = model.predict(X_test_b)
sk_pred = sk_model.predict(X_test_b)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

sk_mse = mean_squared_error(y_test, sk_pred)
sk_r2 = r2_score(y_test, sk_pred)

print("SK-model vs my-model   mse:", sk_mse, mse)
print("SK-model vs my-model   r2:", sk_r2, r2)



