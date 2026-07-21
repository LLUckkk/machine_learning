import numpy as np

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt


from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class BaseBPNetwork:
    """
    单隐层前馈神经网络
    隐藏层和输出层都使用 Sigmoid 激活函数，
    损失函数使用均方误差：
        E = 1/2 * (y_pred - y)^2
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        lr: float = 0.1,
        epochs: int = 1000,
        random_state: int = 42,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.epochs = epochs

        # 随机数生成器构造函数
        rng = np.random.default_rng(random_state)

        # 输入层到隐藏层的权重
        self.W1 = rng.normal(
            loc=0.0,
            scale=0.5,
            size=(input_size, hidden_size),
        )

        # 隐藏层偏置
        self.b1 = np.zeros((1, hidden_size))

        # 隐藏层到输出层的权重
        self.W2 = rng.normal(
            loc=0.0,
            scale=0.5,
            size=(hidden_size, 1)
        )

        # 输出层偏置
        self.b2 = np.zeros((1, 1))

        self.loss_history = []

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """
        sigmoid激活函数
        """
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(output: np.ndarray) -> np.ndarray:
        """
        sigmoid的导数
        这里传入的 output 已经是 sigmoid(z)，因此：
            sigmoid'(z) = output * (1 - output)
        """
        return output * (1 - output)

    def forward(self, X: np.ndarray):
        """
        前向传播
        """
        # 输入层输入至隐藏层
        hidden_input = X.dot(self.W1) + self.b1

        # 隐藏层输入至输出层
        hidden_output = self.sigmoid(hidden_input)

        # 输出层接收隐藏层输入
        final_input = hidden_output.dot(self.W2) + self.b2

        # 输出层激活输出
        final_output = self.sigmoid(final_input)

        return hidden_output, final_output

    @staticmethod
    def mse_loss(
            y_true: np.ndarray,
            y_pred: np.ndarray,
    ) -> float:
        """
        计算均方误差
        """
        return float(
            0.5 * np.mean((y_pred - y_true) ** 2)
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        输出属于正类的概率
        """
        _, output = self.forward(X)
        return output

    def predict(self, X: np.ndarray) -> np.ndarray:
        """阈值为0.5判断正反例"""
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """计算准确率"""
        y = y.reshape(-1, 1)
        y_pred = self.predict(X)

        return float(np.mean(y_pred == y))


class StandardBPNetwork(BaseBPNetwork):
    """
    标准BP算法
    每处理一个样本，就立即进行一次参数更新
    """
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        shuffle: bool = True,
        random_state: int = 42,
    ):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        rng = np.random.default_rng(random_state)
        n_samples = len(X)

        for epoch in range(self.epochs):
            indices = np.arange(n_samples) # 创建等差数列数组

            if shuffle:
                rng.shuffle(indices)

            for index in indices:
                x_i = X[index:index + 1] # 保持原维度
                y_i = y[index:index + 1]

                # ----------前向传播---------
                hidden_output, final_output = self.forward(x_i)

                # ----------反向传播---------
                # 输出层误差信号
                #
                # 损失：
                # E = 1/2 * (y_hat - y)^2
                #
                # 对输出层线性输入求导：
                # delta_output
                # = (y_hat - y) * sigmoid'(z)
                output_delta = (
                    (final_output - y_i) * self.sigmoid_derivative(final_output)
                )

                # 隐藏层误差信号
                hidden_delta = (
                    output_delta.dot(self.W2.T)
                ) * self.sigmoid_derivative(hidden_output)

                # ----------计算梯度---------
                grad_W2 = hidden_output.T.dot(output_delta)
                grad_b2 = output_delta

                grad_W1 = x_i.T.dot(hidden_delta)
                grad_b1 = hidden_delta

                # ----------更新参数---------
                self.W2 -= self.lr * grad_W2
                self.b2 -= self.lr * grad_b2
                self.b1 -= self.lr * grad_b1
                self.W1 -= self.lr * grad_W1

            # 一个epoch结束后计算训练集的损失
            _, train_output = self.forward(X)
            loss = self.mse_loss(y, train_output)
            self.loss_history.append(loss)

        return self


class AccumulatedBPNetwork(BaseBPNetwork):
    """
    累计BP算法
    计算整个训练集上的平均梯度
    每个epoch结束后更新参数
    """
    def fit(
      self,
      X: np.ndarray,
      y: np.ndarray,
    ):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        n_samples = len(X)

        for epoch in range(self.epochs):
            hidden_output, final_output = self.forward(X)

            output_delta = (
                (final_output - y)
                * self.sigmoid_derivative(final_output)
            )

            hidden_delta = (
                output_delta @ self.W2.T
            ) * self.sigmoid_derivative(hidden_output)

            # 更新梯度
            grad_W2 = (
                hidden_output.T.dot(output_delta)
            ) / n_samples

            grad_b2 = np.mean(
                output_delta,
                axis=0,
                keepdims=True,
            )

            grad_W1 = (
                X.T.dot(hidden_delta)
            ) / n_samples

            grad_b1 = np.mean(
                hidden_delta,
                axis=0,
                keepdims=True,
            )

            # 每轮更新一次
            self.W2 -= self.lr * grad_W2
            self.b2 -= self.lr * grad_b2

            self.W1 -= self.lr * grad_W1
            self.b1 -= self.lr * grad_b1

            loss = self.mse_loss(y, final_output)
            self.loss_history.append(loss)

        return self


if __name__ == '__main__':
    # 加载数据
    X, y = make_moons(
        n_samples=500,
        noise=0.20,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y  # 按照y标签的比例分层抽样划分数据集
    )

    # 创建两个网络
    standard_bp = StandardBPNetwork(
        input_size=2,
        hidden_size=8,
        lr=0.1,
        epochs=2000,
        random_state=42,
    )  # standard bp的测试结果更好一些

    accumulated_bp = AccumulatedBPNetwork(
        input_size=2,
        hidden_size=8,
        lr=0.5,  # 累计bp每个epoch只更新一次，所以学习率要大一点
        # 提高lr对累计bp的正确率提高不是特别有帮助，反耳还会使效果降低
        epochs=2000,
        random_state=42,
    )

    # 训练
    standard_bp.fit(
        X_train,
        y_train,
        shuffle=True,
        random_state=42,
    )

    accumulated_bp.fit(
        X_train,
        y_train,
    )

    # 比较准确率
    print("标准 BP：")
    print(
        f"训练集准确率："
        f"{standard_bp.score(X_train, y_train):.4f}"
    )
    print(
        f"测试集准确率："
        f"{standard_bp.score(X_test, y_test):.4f}"
    )
    print(
        f"最终训练损失："
        f"{standard_bp.loss_history[-1]:.6f}"
    )

    print()

    print("累计 BP：")
    print(
        f"训练集准确率："
        f"{accumulated_bp.score(X_train, y_train):.4f}"
    )
    print(
        f"测试集准确率："
        f"{accumulated_bp.score(X_test, y_test):.4f}"
    )
    print(
        f"最终训练损失："
        f"{accumulated_bp.loss_history[-1]:.6f}"
    )

    # 绘制损失曲线
    plt.figure(figsize=(8, 5))

    plt.plot(
        standard_bp.loss_history,
        label="Standard BP",
    )

    plt.plot(
        accumulated_bp.loss_history,
        label="Accumulated BP",
    )

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Standard BP vs Accumulated BP")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


"""
epoch=2000
标准 BP：
训练集准确率：0.9867
测试集准确率：0.9840
最终训练损失：0.007286

累计 BP：
训练集准确率：0.8613
测试集准确率：0.8560
最终训练损失：0.044902

lr=0.9
标准 BP：
训练集准确率：0.9867
测试集准确率：0.9840
最终训练损失：0.007286

累计 BP：
训练集准确率：0.8613
测试集准确率：0.8400
最终训练损失：0.044784

lr = 2.0
标准 BP：
训练集准确率：0.9867
测试集准确率：0.9840
最终训练损失：0.007286

累计 BP：
训练集准确率：0.8613
测试集准确率：0.8400
最终训练损失：0.044584
"""