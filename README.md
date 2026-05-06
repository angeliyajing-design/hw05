# hw05
```markdown
# MNIST手写数字识别 CNN与LeNet-5 深度学习实验
## 项目简介
本实验包含两大任务：
1. 任务一：搭建简易卷积神经网络CNN，完成MNIST手写数字分类识别
2. 任务二：实现经典LeNet-5网络结构，与简易CNN进行精度、速度对比分析

## 文件结构
```
├── cnn_mnist.py        # 任务一 简易CNN模型训练
├── lenet5_mnist.py     # 任务二 LeNet-5模型训练
├── requirements.txt    # 项目环境依赖
├── debug_notes.md      # 运行报错与调试记录
├── report.md           # 完整实验分析报告
└── README.md           # 项目说明文档
```

## 环境配置
Python 3.9及以上
安装依赖命令：
```bash
pip install -r requirements.txt
```

## 数据集
使用MNIST公开手写数字数据集
程序运行后**自动下载**至 ./data 文件夹
无需手动下载、整理文件

## 运行方法
1. 运行任务一CNN
```bash
python cnn_mnist.py
```
2. 运行任务二LeNet-5
```bash
python lenet5_mnist.py
```

## 实验结果
- 简易CNN测试准确率：98.5%左右
- LeNet-5测试准确率：98.8%左右
- 自动绘制损失曲线、可视化识别效果
- 默认CPU训练，支持GPU加速训练

## 网络说明
- CNN：两层卷积+池化+全连接，轻量化入门网络
- LeNet-5：经典初代卷积神经网络，结构标准、参数量合理
- 优化器：Adam
- 损失函数：交叉熵损失CrossEntropyLoss
