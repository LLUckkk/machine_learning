import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class LDA:
    def __init__(self):
        self.w = None
        self.threshold = None
        self.m1 = None # 投影后的中心点
        self.m0 = None

    def fit(self, X, y):
        # 计算类别均值
        mu0 = X[y == 0].mean(axis=0)  # 对所有y值为0的行按照列分别求平均值
        mu1 = X[y == 1].mean(axis=0)

        # 计算类内散度矩阵
        Sw = np.zeros((X.shape[1], X.shape[1]))
        for x in X[y == 0]:
            diff = (x - mu0).reshape(-1, 1)
            Sw += diff.dot(diff.T)

        for x in X[y == 1]:
            diff = (x - mu1).reshape(-1, 1)
            Sw += diff.dot(diff.T)

        # 根据类内散度矩阵得到投影方向
        self.w = np.linalg.pinv(Sw).dot(mu0 - mu1)

        # 根据均值得到threshold值
        self.m0 = self.w.dot(mu0)
        self.m1 = self.w.dot(mu1)

    def predict(self, X):
        score = X.dot(self.w)

        dist0 = np.abs(score - self.m0)
        dist1 = np.abs(score - self.m1)

        return (
                dist1 < dist0
        ).astype(int)


# 西瓜数据集3.0α
X = [
    [0.697, 0.460],
    [0.774, 0.376],
    [0.634, 0.264],
    [0.608, 0.318],
    [0.403, 0.237],
    [0.481, 0.149],
    [0.437, 0.211],
    [0.666, 0.091],
    [0.243, 0.267],
    [0.245, 0.057],
    [0.343, 0.099],
    [0.639, 0.161],
    [0.657, 0.198],
    [0.360, 0.370],
    [0.593, 0.042]
]
y = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
X = np.array(X)
y = np.array(y)
print(X.shape, y.shape)

model_watermelon = LDA()
model_watermelon.fit(X, y)

X_test = [
    [0.556, 0.215],
    # [0.593, 0.042]
    [0.719, 0.103],
]
y_test = [1, 0]
X_test = np.array(X_test)
y_test = np.array(y_test)

y_pred = model_watermelon.predict(X_test)
true = 0
for i in range(y_pred.shape[0]):
    print(y_pred[i])
    if y_pred[i] == y_test[i]:
        true = true + 1

accuracy = true/y_pred.shape[0]
print('Accuracy on test set: {:.2f}%'.format(100 * accuracy))
# LDA的accuracy竟然100吗。




# # 加载数据
# data = load_iris()
# X = data.data
# y = data.target
# # 只做二分类，取类别0和1
# mask = y != 2
# X = X[mask]
# y = y[mask]
#
# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#
# # 标准化特征
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
#
# # 训练模型
# model = LDA()
# model.fit(X_train, y_train)
#
# # 预测
# y_pred = model.predict(X_test)
# true = 0
# for i in range(y_pred.shape[0]):
#     if y_pred[i] == y_test[i]:
#         true += 1
#
# accuracy = true/y_pred.shape[0]
# print('Accuracy on test set: {:.2f}%'.format(100 * accuracy))
