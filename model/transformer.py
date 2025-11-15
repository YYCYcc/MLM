import os
import torch
import torch.nn as nn
#from transformers import pipeline

class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TrafficTransformer, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.transformer = nn.Transformer(d_model=hidden_size, num_encoder_layers=6, nhead=8)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)  # 输入特征进行嵌入
        x = x.unsqueeze(0)  # 加一个 batch 维度
        x = self.transformer(x, x)  # Transformer 进行处理
        x = x.squeeze(0)  # 移除 batch 维度
        x = self.fc(x)  # 输出预测
        return x