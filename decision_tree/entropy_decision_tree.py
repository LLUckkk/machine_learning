from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class TreeNode:
    """
    决策树节点。

    内部节点保存：
    - feature_index：划分特征下标
    - threshold：划分阈值
    - left：左子树
    - right：右子树

    叶节点主要保存：
    - prediction：预测类别
    """

    feature_index: Optional[int] = None
    threshold: Optional[float] = None

    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None

    prediction: Any = None
    gain: float = 0.0

    @property
    def is_leaf(self) -> bool:
        """左右子树都不存在时，当前节点就是叶节点。"""
        return self.left is None and self.right is None


class EntropyDecisionTreeClassifier:
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_gain: float = 1e-12,
    ):
        """
        max_depth：
            树的最大深度。None 表示不限制。

        min_samples_split：
            节点至少包含多少个样本，才允许继续划分。

        min_samples_leaf：
            划分后每个叶节点至少包含多少个样本。

        min_gain：
            信息增益至少达到多少，才接受本次划分。
        """

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain

        self.root: Optional[TreeNode] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
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
            raise ValueError("训练集不能为空")

        self.root = self._build_tree(X, y, depth=0)

        return self

    @staticmethod
    def _entropy(y: np.ndarray) -> float:
        """
        计算标签集合的信息熵。

        H(D) = -Σ p_k log2(p_k)
        """

        if len(y) == 0:
            return 0.0

        _, counts = np.unique(y, return_counts=True)

        probabilities = counts / len(y)

        return float(
            -np.sum(
                probabilities * np.log2(probabilities)
            )
        )

    @staticmethod
    def _majority_class(y: np.ndarray):
        """返回当前样本中数量最多的类别。"""

        classes, counts = np.unique(
            y,
            return_counts=True,
        )

        majority_index = np.argmax(counts)

        return classes[majority_index]

    def _find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        """
        枚举所有特征和候选阈值，
        找到信息增益最大的划分。
        """

        n_samples, n_features = X.shape

        parent_entropy = self._entropy(y)

        best_gain = -np.inf
        best_split = None

        # 枚举每一个特征
        for feature_index in range(n_features):
            feature_values = X[:, feature_index]

            # 去重并排序
            unique_values = np.unique(feature_values)

            # 只有一个取值时，该特征无法划分
            if len(unique_values) < 2:
                continue

            # 候选阈值取相邻取值的中点
            thresholds = (
                unique_values[:-1] + unique_values[1:]
            ) / 2

            # 枚举该特征的所有候选阈值
            for threshold in thresholds:
                left_mask = feature_values <= threshold  # 这里注意feature_values是个数组
                # 得到的mask是一个布尔数组
                right_mask = feature_values > threshold

                left_count = np.sum(left_mask)
                right_count = np.sum(right_mask)

                # 保证左右子节点都有足够样本
                if left_count < self.min_samples_leaf:
                    continue

                if right_count < self.min_samples_leaf:
                    continue

                left_y = y[left_mask]  # 取出left_mask中true所在位置对应的元素
                right_y = y[right_mask]

                # 划分后的加权熵
                children_entropy = (
                    left_count / n_samples
                    * self._entropy(left_y)
                    +
                    right_count / n_samples
                    * self._entropy(right_y)
                )

                # 信息增益
                information_gain = (
                    parent_entropy - children_entropy
                )

                if information_gain > best_gain:
                    best_gain = information_gain

                    best_split = {
                        "feature_index": feature_index,
                        "threshold": float(threshold),
                        "left_mask": left_mask,
                        "right_mask": right_mask,
                        "gain": float(information_gain),
                    }

        return best_split

    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int,
    ) -> TreeNode:
        """
        递归构建决策树。
        """

        majority_class = self._majority_class(y)

        # 停止条件1：当前节点样本属于同一类别
        if len(np.unique(y)) == 1:
            return TreeNode(
                prediction=majority_class
            )

        # 停止条件2：达到最大深度
        if (
            self.max_depth is not None
            and depth >= self.max_depth
        ):
            return TreeNode(
                prediction=majority_class
            )

        # 停止条件3：样本数量太少
        if len(y) < self.min_samples_split:
            return TreeNode(
                prediction=majority_class
            )

        best_split = self._find_best_split(X, y)

        # 停止条件4：找不到有效划分
        if best_split is None:
            return TreeNode(
                prediction=majority_class
            )

        # 停止条件5：最佳信息增益太小
        if best_split["gain"] <= self.min_gain:
            return TreeNode(
                prediction=majority_class
            )

        left_mask = best_split["left_mask"]
        right_mask = best_split["right_mask"]

        # 递归构建左右子树
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
            feature_index=best_split["feature_index"],
            threshold=best_split["threshold"],
            left=left_tree,
            right=right_tree,
            prediction=majority_class,
            gain=best_split["gain"],
        )

    def _predict_one(self, x: np.ndarray):
        """预测一个样本。"""

        if self.root is None:
            raise RuntimeError("模型尚未训练，请先调用 fit")

        node = self.root

        # 从根节点一路走到叶节点
        while not node.is_leaf:
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right

        return node.prediction

    def predict(self, X: np.ndarray) -> np.ndarray:
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
        """计算准确率。"""

        y = np.asarray(y)
        y_pred = self.predict(X)

        return float(np.mean(y_pred == y))

    def print_tree(
        self,
        feature_names=None,
    ):
        """以文本形式打印决策树。"""

        if self.root is None:
            raise RuntimeError("模型尚未训练，请先调用 fit")

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
                f"{indent}预测类别：{node.prediction}"
            )
            return

        if feature_names is None:
            feature_name = f"特征{node.feature_index}"
        else:
            feature_name = feature_names[
                node.feature_index
            ]

        print(
            f"{indent}若 {feature_name} "
            f"<= {node.threshold:.4f} "
            f"(信息增益={node.gain:.4f})"
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
    # 两个特征分别是：密度、含糖率

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

    model = EntropyDecisionTreeClassifier(
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
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