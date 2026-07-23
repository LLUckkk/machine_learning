import os
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MLP(nn.Module):  # pytorch中，自定义神经网络通常都要继承nn.Module
    """
    使用多层感知机识别 MNIST 手写数字。
    输入图像：
    (batch_size, 1, 28, 28)
    展开后：
    (batch_size, 784)
    网络结构：
    784 -> 256 -> 128 -> 10
    """

    def __init__(self) -> None:
        super().__init__()

        self.network = nn.Sequential( # 输入按照从上到下的顺序依次经过这些网络层
            nn.Flatten(),  # 经过flatten之后图片从二维变成一维了
            # 现在是64, 28*28.其中64是batch_size

            nn.Linear(
                in_features=28 * 28,
                out_features=256,
            ),
            # nn.ReLU(),
            nn.Sigmoid(),

            nn.Linear(
                in_features=256,
                out_features=128,
            ),
            # nn.ReLU(),
            nn.Sigmoid(),

            nn.Linear(
                in_features=128,
                out_features=10,
            ),
        )

    # 该函数定义数据如何在网络中传播。比如: model(image),实际调用的是model.forward(image)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        返回10个类别对应得分
        """
        return self.network(x)


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """训练一个epoch"""
    model.train()  # 网络切换到训练模式，当有dropout的时候一定要写

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in data_loader:  # dataloader把数据分批送入网络，这里表示一个批次
        images = images.to(device)
        labels = labels.to(device)  # 数据也必须迁移到同一个设备：

        optimizer.zero_grad()  # 一个batch是指使用这个batch计算累计梯度更新参数
        # 因此每个batch开始之前先清空梯度

        logits = model(images)  # 前向传播

        loss = criterion(logits, labels)  # 计算损失

        loss.backward()  # 反向传播计算梯度，存入parameter.grad

        optimizer.step()  # 读取parameter.grad，更新参数

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size  # loss.item把只包含一个数的 Tensor 转换成 Python 数值

        predictions = torch.argmax(logits, dim=1)

        correct += (predictions == labels).sum().item()
        total += batch_size

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


@torch.no_grad()   # 整个函数中关闭梯度记录
def evaluate(
        model: nn.Module,
        data_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device
) -> float:
    """评估模型"""
    model.eval()  # 模型切换到评估模式

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

        predictions = torch.argmax(logits, dim=1)

        correct += (predictions == labels).sum().item()
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
    """查看部分测试样本的预测结果"""

    model.eval()

    images, labels = next(iter(data_loader))

    images = images[:count].to(device)
    labels = labels[:count].to(device)

    logits = model(images)

    probabilities = torch.softmax(logits, dim=1)

    predictions = torch.argmax(probabilities, dim=1)

    confidences = torch.max(logits, dim=1).values

    for index in range(count):
        print(
            f"样本{index + 1}: "
            f"真实标签={labels[index].item()}, "
            f"预测标签={predictions[index].item()}, "
            f"置信度={confidences[index].item():.4f}"
        )


def main() -> None:
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("使用设备：", device)

    transform = transforms.Compose([  # 按照顺序执行多个数据变换
        transforms.ToTensor(),  # 图片转换为pytorch tensor，像素从0-255转为0-1
        transforms.Normalize(  # 对像素进行标准化
            mean=(0.1307,),
            std=(0.3081,),
        ),
    ])

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        transform=transform,
        download=True,
    )  # 加载训练集

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,  # 表示加载测试集
        transform=transform,
        download=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0, # 表示由主进程加载数据
    )  # Dataloader的作用是：把数据集分批送入网络

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    model = MLP().to(device)  # 模型放到对应设备，数据也必须迁移到同一个单位

    criterion = nn.CrossEntropyLoss()  # 多分类任务使用交叉熵损失函数

    optimizer = torch.optim.Adam(
        model.parameters(),  # 返回模型中所有需要训练的参数
        lr=0.001,
    )  # 优化器利用梯度更新参数

    epochs = 10

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

    model_path = "mlp_mnist.pth"

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