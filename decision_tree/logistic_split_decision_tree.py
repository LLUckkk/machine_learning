from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class TreeNode:
    """
    对数几率回归决策树的节点。

    内部节点保存：
    - weights：对数几率回归的权重
    - bias：偏置项
    - mean、scale：该节点训练时使用的标准化参数
    - left、right：左右子树

    叶节点保存：
    - prediction：预测类别
    """

    weights: Optional[np.ndarray] = None
    bias: float = 0.0

    mean: Optional[np.ndarray] = None
    scale: Optional[np.ndarray] = None

    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None

    prediction: Any = None

    split_gini: float = 0.0
    gini_decrease: float = 0.0
    n_samples: int = 0

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class LogisticSplitDecisionTreeClassifier:
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 1e-12,
        lr: float = 0.1,  # 对树几率回归训练的学习率
        epochs: int = 5000,
        l2: float = 1e-3,  # 正则化系数
        tol: float = 1e-8,  # 梯度足够小的时候提前停止训练
    ):
        """
        决策树参数：

        max_depth：
            最大树深，None 表示不限制。

        min_samples_split：
            节点至少包含多少个样本才允许继续划分。

        min_samples_leaf：
            左右子节点至少分别包含多少个样本。

        min_impurity_decrease：
            基尼指数至少下降多少才接受本次划分。

        对数几率回归参数：

        lr：
            梯度下降学习率。

        epochs：
            最大训练轮数。

        l2：
            L2 正则化系数。

        tol：
            梯度足够小时提前停止训练。
        """

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease

        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.tol = tol

        self.root: Optional[TreeNode] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        """训练决策树。"""

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X 必须是二维数组")

        if y.ndim != 1:
            raise ValueError("y 必须是一维数组")

        if len(X) != len(y):
            raise ValueError("X 和 y 的样本数量必须相同")

        if len(X) == 0:
            raise ValueError("训练数据不能为空")

        self.classes_ = np.unique(y)

        if len(self.classes_) != 2:
            raise ValueError(
                "当前版本只支持二分类任务"
            )

        self.root = self._build_tree(
            X,
            y,
            depth=0,
        )

        return self

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """计算 Sigmoid，并限制数值范围避免溢出。"""

        z = np.clip(z, -500, 500) # 设置数据范围

        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _gini(y: np.ndarray) -> float:
        """计算标签集合的基尼指数。"""

        if len(y) == 0:
            return 0.0

        _, counts = np.unique(
            y,
            return_counts=True,
        )

        probabilities = counts / len(y)

        return float(
            1.0 - np.sum(probabilities ** 2)
        )

    @staticmethod
    def _majority_class(y: np.ndarray):
        """返回当前节点中的多数类别。"""

        classes, counts = np.unique(
            y,
            return_counts=True,
        )

        return classes[np.argmax(counts)]

    def _fit_logistic_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        """
        在当前节点训练一个对数几率回归模型。

        返回：
        - 权重 weights
        - 偏置 bias
        - 标准化参数 mean、scale
        - 左右子节点的布尔掩码
        """

        n_samples, n_features = X.shape

        # 在当前节点中对特征进行标准化
        mean = np.mean(X, axis=0)
        scale = np.std(X, axis=0)

        # 防止某个特征标准差为0
        scale = np.where(
            scale < 1e-12,
            1.0,
            scale,
        )

        X_scaled = (X - mean) / scale

        # 将原始类别转换为0和1
        y_binary = (
            y == self.classes_[1]
        ).astype(float).reshape(-1, 1)

        weights = np.zeros(
            (n_features, 1)
        )

        bias = 0.0

        for _ in range(self.epochs):
            # 线性部分
            z = X_scaled @ weights + bias

            # 当前属于类别1的概率
            probabilities = self._sigmoid(z)

            # 交叉熵损失对权重的梯度
            weight_gradient = (
                X_scaled.T
                @ (probabilities - y_binary)
                / n_samples
            )

            # 加入L2正则化
            weight_gradient += (
                self.l2 * weights
            )

            # 偏置项梯度
            bias_gradient = float(
                np.mean(
                    probabilities - y_binary
                )
            )

            # 梯度下降
            weights -= (
                self.lr * weight_gradient
            )

            bias -= (
                self.lr * bias_gradient
            )

            # 梯度足够小时提前结束
            gradient_size = (
                np.linalg.norm(weight_gradient)
                + abs(bias_gradient)
            )

            if gradient_size < self.tol:
                break

        # Logistic Regression中：
        #
        # probability <= 0.5
        # 等价于 z <= 0
        #
        # probability > 0.5
        # 等价于 z > 0

        scores = (
            X_scaled @ weights + bias
        ).reshape(-1)

        left_mask = scores <= 0
        right_mask = scores > 0

        return {
            "weights": weights.reshape(-1),
            "bias": float(bias),
            "mean": mean,
            "scale": scale,
            "left_mask": left_mask,
            "right_mask": right_mask,
        }

    def _find_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        """
        训练对数几率回归模型，
        使用它产生的超平面划分当前节点。
        """

        split = self._fit_logistic_split(
            X,
            y,
        )

        left_mask = split["left_mask"]
        right_mask = split["right_mask"]

        left_count = np.sum(left_mask)
        right_count = np.sum(right_mask)

        # 如果一侧样本数量太少，划分无效
        if left_count < self.min_samples_leaf:
            return None

        if right_count < self.min_samples_leaf:
            return None

        left_y = y[left_mask]
        right_y = y[right_mask]

        parent_gini = self._gini(y)

        # 计算划分后的加权基尼指数
        children_gini = (
            left_count / len(y)
            * self._gini(left_y)
            +
            right_count / len(y)
            * self._gini(right_y)
        )

        split["split_gini"] = float(
            children_gini
        )

        split["gini_decrease"] = float(
            parent_gini - children_gini
        )

        return split

    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int,
    ) -> TreeNode:
        """递归构建决策树。"""

        majority_class = self._majority_class(y)

        # 停止条件1：当前节点已经完全纯净
        if len(np.unique(y)) == 1:
            return TreeNode(
                prediction=majority_class,
                n_samples=len(y),
            )

        # 停止条件2：达到最大深度
        if (
            self.max_depth is not None
            and depth >= self.max_depth
        ):
            return TreeNode(
                prediction=majority_class,
                n_samples=len(y),
            )

        # 停止条件3：样本数太少
        if len(y) < self.min_samples_split:
            return TreeNode(
                prediction=majority_class,
                n_samples=len(y),
            )

        split = self._find_split(X, y)

        # 停止条件4：对数几率回归无法产生有效划分
        if split is None:
            return TreeNode(
                prediction=majority_class,
                n_samples=len(y),
            )

        # 停止条件5：不纯度下降得太少
        if (
            split["gini_decrease"]
            <= self.min_impurity_decrease
        ):
            return TreeNode(
                prediction=majority_class,
                n_samples=len(y),
            )

        left_mask = split["left_mask"]
        right_mask = split["right_mask"]

        left_tree = self._build_tree(
            X[left_mask],
            y[left_mask],
            depth + 1,
        )

        right_tree = self._build_tree(
            X[right_mask],
            y[right_mask],
            depth + 1,
        )

        return TreeNode(
            weights=split["weights"],
            bias=split["bias"],
            mean=split["mean"],
            scale=split["scale"],
            left=left_tree,
            right=right_tree,
            prediction=majority_class,
            split_gini=split["split_gini"],
            gini_decrease=split[
                "gini_decrease"
            ],
            n_samples=len(y),
        )

    def _predict_one(
        self,
        x: np.ndarray,
    ):
        """预测一个样本。"""

        if self.root is None:
            raise RuntimeError(
                "模型尚未训练，请先调用 fit"
            )

        node = self.root

        while not node.is_leaf:
            x_scaled = (
                x - node.mean
            ) / node.scale

            score = (
                x_scaled @ node.weights
                + node.bias
            )

            if score <= 0:
                node = node.left
            else:
                node = node.right

        return node.prediction

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """预测多个样本。"""

        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        predictions = [
            self._predict_one(x)
            for x in X
        ]

        return np.asarray(predictions)

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """计算分类准确率。"""

        y = np.asarray(y)
        y_pred = self.predict(X)

        return float(
            np.mean(y_pred == y)
        )

    def print_tree(
        self,
        feature_names=None,
    ):
        """打印决策树结构。"""

        if self.root is None:
            raise RuntimeError(
                "模型尚未训练，请先调用 fit"
            )

        self._print_node(
            self.root,
            depth=0,
            feature_names=feature_names,
        )

    def _print_node(
        self,
        node: TreeNode,
        depth: int,
        feature_names,
    ):
        indent = "    " * depth

        if node.is_leaf:
            print(
                f"{indent}预测类别："
                f"{node.prediction} "
                f"(样本数={node.n_samples})"
            )
            return

        if feature_names is None:
            feature_names = [
                f"特征{i}"
                for i in range(
                    len(node.weights)
                )
            ]

        # 将标准化空间中的方程转换回原始特征空间
        #
        # Σ w_j * ((x_j - mean_j) / scale_j) + b = 0
        #
        # 等价于：
        #
        # Σ (w_j / scale_j) * x_j
        # + b'
        # = 0

        original_weights = (
            node.weights / node.scale
        )

        original_bias = (
            node.bias
            - np.sum(
                node.weights
                * node.mean
                / node.scale
            )
        )

        terms = []

        for weight, feature_name in zip(
            original_weights,
            feature_names,
        ):
            terms.append(
                f"{weight:+.4f}×{feature_name}"
            )

        equation = " ".join(terms)

        equation += (
            f" {original_bias:+.4f}"
        )

        print(
            f"{indent}若 {equation} <= 0 "
            f"(划分后Gini={node.split_gini:.4f}, "
            f"Gini下降={node.gini_decrease:.4f})"
        )

        print(f"{indent}├── 是：")
        self._print_node(
            node.left,
            depth + 1,
            feature_names,
        )

        print(f"{indent}└── 否：")
        self._print_node(
            node.right,
            depth + 1,
            feature_names,
        )


if __name__ == "__main__":
    # 西瓜数据集3.0α
    # 特征：密度、含糖率

    X_train = np.array([
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
        [0.593, 0.042],
    ])

    y_train = np.array([
        1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0,
    ])

    X_test = np.array([
        [0.556, 0.215],
        [0.719, 0.103],
    ])

    y_test = np.array([1, 0])

    model = LogisticSplitDecisionTreeClassifier(
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=1e-12,
        lr=0.1,
        epochs=5000,
        l2=1e-3,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("预测结果：", y_pred)
    print("真实结果：", y_test)

    print(
        "训练集准确率：",
        model.score(X_train, y_train),
    )

    print(
        "测试集准确率：",
        model.score(X_test, y_test),
    )

    print("\n决策树结构：")

    model.print_tree(
        feature_names=["密度", "含糖率"]
    )