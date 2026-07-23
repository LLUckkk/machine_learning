import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def set_seed(seed: int = 42) -> None:
    """固定随机种子，方便复现实验结果。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LeNet(nn.Module):
    """
    一个经典的 LeNet 风格卷积神经网络。

    输入：
        (batch_size, 1, 28, 28)
        这里的1表示1个灰度通道 channel
        灰度图每个位置只有一个数表示像素颜色，所以是一个通道
        彩色图一般有三个通道，RGB

        cnn并不像mlp一样直接将二维展开，而是先保留二维结构

    网络结构：
        卷积 -> ReLU -> 池化
        卷积 -> ReLU -> 池化
        全连接 -> ReLU
        全连接 -> 10个类别得分
    """

    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # 输入：(N, 1, 28, 28)
            nn.Conv2d(
                in_channels=1,
                out_channels=6,  # 6个卷积核，自己指定outchannels
                kernel_size=5,  # 5*5的卷积核，遍历图片矩阵，对应位置元素相乘并相加，得到一个特征map
                stride=1,
                padding=2,
            ),
            # 输出：(N, 6, 28, 28) 得到6张特征图，此时六个特征图同样的位置共同决定了原图对应位置的像素，所以变成了6个通道
            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
            # 输出：(N, 6, 14, 14) 每个2*2区域压缩为取最大值

            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=0,
            ),
            # 输出：(N, 16, 10, 10)
            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
            # 输出：(N, 16, 5, 5)
        )

        self.classifier = nn.Sequential(
            # 16 × 5 × 5 = 400
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),

            nn.Linear(120, 84),
            nn.ReLU(),

            nn.Linear(84, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        返回值不是概率，而是10个类别对应的原始得分 logits。
        """
        x = self.features(x)

        # 从四维特征图展开为二维矩阵
        # (N, 16, 5, 5) -> (N, 400)
        x = torch.flatten(x, start_dim=1)

        logits = self.classifier(x)

        return logits


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """训练一个 epoch。"""

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        # 清空上一次反向传播留下的梯度
        optimizer.zero_grad()

        # 前向传播
        logits = model(images)

        # 计算交叉熵损失
        loss = criterion(logits, labels)

        # 反向传播，计算梯度
        loss.backward()

        # 更新参数
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """在测试集上评估模型。"""

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


@torch.no_grad()
def show_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    count: int = 10,
) -> None:
    """打印一批测试样本的预测结果。"""

    model.eval()

    images, labels = next(iter(data_loader))

    images = images[:count].to(device)
    labels = labels[:count].to(device)

    logits = model(images)

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    predictions = torch.argmax(
        probabilities,
        dim=1,
    )

    confidences = torch.max(
        probabilities,
        dim=1,
    ).values

    for index in range(count):
        print(
            f"样本{index + 1}: "
            f"真实标签={labels[index].item()}, "
            f"预测标签={predictions[index].item()}, "
            f"置信度={confidences[index].item():.4f}"
        )


def main() -> None:
    set_seed(42)

    # 自动选择 GPU 或 CPU
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("使用设备：", device)

    # MNIST原始像素范围为[0, 1]
    # Normalize将其近似标准化
    transform = transforms.Compose([
        transforms.ToTensor(),

        transforms.Normalize(
            mean=(0.1307,),
            std=(0.3081,),
        ),
    ])

    data_root = "./data"

    train_dataset = datasets.MNIST(
        root=data_root,
        train=True,
        transform=transform,
        download=True,
    )

    test_dataset = datasets.MNIST(
        root=data_root,
        train=False,
        transform=transform,
        download=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    model = LeNet().to(device)

    # CrossEntropyLoss内部已经完成了
    # LogSoftmax + Negative Log Likelihood
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    epochs = 5

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        test_loss, test_accuracy = evaluate(
            model=model,
            data_loader=test_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"训练损失={train_loss:.4f}, "
            f"训练准确率={train_accuracy:.4%}, "
            f"测试损失={test_loss:.4f}, "
            f"测试准确率={test_accuracy:.4%}"
        )

    # 保存模型参数
    model_path = "lenet_mnist.pth"

    torch.save(
        model.state_dict(),
        model_path,
    )

    print(
        "模型已保存到：",
        os.path.abspath(model_path),
    )

    print("\n部分测试样本预测结果：")

    show_predictions(
        model=model,
        data_loader=test_loader,
        device=device,
        count=10,
    )


if __name__ == "__main__":
    main()