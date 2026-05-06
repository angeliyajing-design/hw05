import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time

# 解决matplotlib中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设备自动选择（优先GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ---------------------- 1. 实现LeNet-5模型 ----------------------
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 卷积层1：输入1通道，输出6通道，5x5卷积核，无padding
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=0)
        self.tanh1 = nn.Tanh()
        # 池化层1：2x2平均池化，步长2
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 卷积层2：输入6通道，输出16通道，5x5卷积核，无padding
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=0)
        self.tanh2 = nn.Tanh()
        # 池化层2：2x2平均池化，步长2
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 全连接层（LeNet-5经典结构）
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 两次池化后特征图为4x4
        self.tanh3 = nn.Tanh()
        self.fc2 = nn.Linear(120, 84)
        self.tanh4 = nn.Tanh()
        self.fc3 = nn.Linear(84, 10)  # 输出10个类别

    def forward(self, x):
        x = self.pool1(self.tanh1(self.conv1(x)))  # 28→24→12
        x = self.pool2(self.tanh2(self.conv2(x)))  # 12→8→4
        x = x.view(-1, 16 * 4 * 4)  # 展平为向量
        x = self.tanh3(self.fc1(x))
        x = self.tanh4(self.fc2(x))
        x = self.fc3(x)
        return x

# ---------------------- 2. 数据加载（和任务一保持一致） ----------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集标准归一化
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 保持和任务一相同的训练设置：batch_size=64
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ---------------------- 3. 训练/测试函数 ----------------------
def train(model, train_loader, criterion, optimizer, device, epochs=5):
    model.train()
    train_losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs}, 训练损失: {epoch_loss:.4f}")
    return train_losses

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = correct.double() / len(test_loader.dataset) * 100
    print(f"\n测试集损失: {test_loss:.4f}, 测试集准确率: {test_acc:.2f}%")
    return test_loss, test_acc

# ---------------------- 4. 主函数（记录训练耗时+输出结果） ----------------------
def main():
    # 初始化模型、损失函数、优化器（和任务一保持一致：Adam、lr=0.001）
    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 记录训练耗时
    start_time = time.time()
    print("开始训练LeNet-5模型...")
    train_losses = train(model, train_loader, criterion, optimizer, device, epochs=5)
    end_time = time.time()
    train_time = end_time - start_time
    print(f"训练总耗时: {train_time:.2f} 秒")

    # 测试模型
    test_loss, test_acc = test(model, test_loader, criterion, device)

    # 保存模型
    torch.save(model.state_dict(), 'lenet5_mnist.pth')
    print("LeNet-5模型已保存为 lenet5_mnist.pth")

    # 绘制训练损失曲线
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='LeNet-5 Training Loss', linewidth=2)
    plt.xlabel('训练轮次 (Epoch)', fontsize=12)
    plt.ylabel('损失值 (Loss)', fontsize=12)
    plt.title('LeNet-5模型训练损失曲线', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # 输出报告用关键数据
    print("\n===== 任务二报告关键数据 =====")
    print(f"训练轮数: 5")
    print(f"批量大小: 64")
    print(f"优化器: Adam")
    print(f"学习率: 0.001")
    print(f"是否使用GPU: {'是' if torch.cuda.is_available() else '否'}")
    print(f"训练耗时: {train_time:.2f} 秒")
    print(f"测试集准确率: {test_acc:.2f}%")

if __name__ == '__main__':
    main()
