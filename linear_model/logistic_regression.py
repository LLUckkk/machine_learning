import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class LogisticRegression:
    def __init__(
        self,
        lr=0.1,
        epochs=10000
    ):
        self.lr = lr
        self.epochs = epochs
        self.w = None

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def fit(self, X, y):
        y = y.reshape(-1, 1)
        self.w = np.zeros((X.shape[1], 1))

        for epoch in range(self.epochs):
            z = X.dot(self.w)
            p = self.sigmoid(z)

            # 梯度
            gradient = X.T.dot(p - y) / len(X)
            # 学习率：沿着梯度下降的方向要走多远
            self.w -= self.lr * gradient

    def predict_proba(self, X):
        return self.sigmoid(np.dot(X, self.w))

    def predict(self, X):
        proba = self.predict_proba(X)

        return (proba > 0.5).astype(int)


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

X_b = np.c_[X, np.ones((X.shape[0], 1))]

model_watermelon = LogisticRegression()
model_watermelon.fit(X_b, y)

X_test = [
    [0.556, 0.215, 1],
    # [0.593, 0.042]
    [0.719, 0.103, 1],
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
# 西瓜数据集太少了，正确率不高

# # 加载数据
# data = load_breast_cancer()
#
# X = data.data
# y = data.target
# # 打印数据形状
# print(X.shape, y.shape)
#
# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 对特征的标准化
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
#
# # 添加偏置项
# # np.c_是按列合并数组
# X_train = np.c_[X_train,  np.ones((len(X_train), 1))]
# X_test = np.c_[X_test,  np.ones((len(X_test), 1))]
#
# # 训练模型
# model = LogisticRegression()
# model.fit(X_train, y_train)
#
# # 测试模型
# y_pred = model.predict(X_test)
# true = 0
# for i in range(y_pred.shape[0]):
#     if y_pred[i] == y_test[i]:
#         true += 1
#
# accuracy = true / y_pred.shape[0]
# print('Accuracy on test set: {:.2f}%'.format(100 * accuracy))







