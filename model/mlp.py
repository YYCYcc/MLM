import torch
import torch.nn as nn


# 定义一个简单的全连接神经网络模型（MLP）
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 第一层：输入到隐藏层
        self.fc2 = nn.Linear(hidden_size, output_size)  # 第二层：隐藏层到输出层

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 激活函数ReLU
        x = self.fc2(x)  # 输出层
        return x
