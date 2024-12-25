# %%
import os
print("Current working directory:", os.getcwd())
import importlib
import sys
sys.path.append('/data')
print(os.listdir('/data'))
import torch
import torch.nn as nn
# LSTM 
from model.lstm import LSTM
# GRU 
from model.gru import GRUModel
# Neural Network 
from model.neural import NeuralODE
# Transformer 
from transformer import TransformerModel
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.model_selection import train_test_split  # 从 scikit-learn 的 model_selection 模块导入 split 方法用于分割训练集和测试集
from sklearn.preprocessing import StandardScaler  
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, classification_report

import torch.optim as optim
from sklearn.metrics import roc_curve, auc

# %%
data = pd.read_csv('/data/data.csv').values
data =  np.random.permutation(data)
print(data)
plt.hist(data[:,0], edgecolor='black')
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.show()
X = data[:,1:]
Y= data[:,0]
print(X)
print(Y)

# %%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).long()
X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).long()
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
val_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# %%


# %%
class TransformerModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TransformerModel, self).__init__()
        # 定义 Transformer 编码器，并指定输入维数和头数
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=1)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        # 定义全连接层，将 Transformer 编码器的输出映射到分类空间
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        # 在序列的第2个维度（也就是时间步或帧）上添加一维以适应 Transformer 的输入格式
        x = x.unsqueeze(1)
        # 将输入数据流经 Transformer 编码器进行特征提取
        x = self.encoder(x)
        # 通过压缩第2个维度将编码器的输出恢复到原来的形状
        x = x.squeeze(1)
        # 将编码器的输出传入全连接层，获得最终的输出结果
        x = self.fc(x)
        return x


# %%
print("创建模型")
# 初始化 Transformer 模型

model = TransformerModel(input_size=1015, num_classes=2)

# %% [markdown]
# 

# %%
# 定义损失函数和优化器

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# 存储训练过程中的损失值和召回率
train_losses = []
recall_scores = []

accuracy_scores = []  # 初始化accuracy_scores
precision_scores = []
f1_scores = []
mse_scores = []

print("训练模型")
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播计算输出结果
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # 反向传播，更新梯度并优化模型参数
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 记录损失值
    train_losses.append(loss.item())

    # 每10个epoch计算召回率
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            _, predicted = torch.max(outputs, 1)
            # 计算各类指标
            accuracy = accuracy_score(y_train, predicted)
            recall = recall_score(y_train, predicted, average='macro')  # 宏平均
            precision = precision_score(y_train, predicted, average='macro')  # 宏平均
            f1 = f1_score(y_train, predicted, average='macro')  # 宏平均
            mse = mean_squared_error(y_train, predicted)  # MSE

            # 记录每个指标的值
            accuracy_scores.append(accuracy)
            recall_scores.append(recall)
            precision_scores.append(precision)
            f1_scores.append(f1)
            mse_scores.append(mse)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}, '
              f'Recall: {recall:.4f}, Precision: {precision:.4f}, F1 Score: {f1:.4f}, MSE: {mse:.4f}')
print("测试模型")
# 测试模型的评分


# %%
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs.data, 1)
    # 计算测试集的各类评估指标
    print(predicted)
    accuracy = accuracy_score(y_test, predicted)
    recall = recall_score(y_test, predicted, average='macro')  # 宏平均
    precision = precision_score(y_test, predicted, average='macro')  # 宏平均
    f1 = f1_score(y_test, predicted, average='macro')  # 宏平均
    mse = mean_squared_error(y_test, predicted)  # MSE

    print(f'Test Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, '
          f'F1 Score: {f1:.4f}, MSE: {mse:.4f}')

    # 输出分类报告，查看每个类别的性能
    print("\nClassification Report:")
    print(classification_report(y_test, predicted))

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid()
    plt.show()

    # 绘制准确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(10, num_epochs + 1, 10), accuracy_scores, label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid()
    plt.show()

    # 绘制召回率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(10, num_epochs + 1, 10), recall_scores, label='Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.title('Recall Curve')
    plt.legend()
    plt.grid()
    plt.show()

    # 绘制精确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(10, num_epochs + 1, 10), precision_scores, label='Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.title('Precision Curve')
    plt.legend()
    plt.grid()
    plt.show()

    # 绘制 F1 分数曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(10, num_epochs + 1, 10), f1_scores, label='F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Curve')
    plt.legend()
    plt.grid()
    plt.show()

    # 绘制 MSE 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(10, num_epochs + 1, 10), mse_scores, label='Mean Squared Error')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title('Mean Squared Error Curve')
    plt.legend()
    plt.grid()
    plt.show()

# %%
outputs = model(X_test)
#print(X_test)
_, predicted = torch.max(outputs.data, 1)


# %%

Y_test=y_test.detach().numpy().reshape(len(y_test),-1)

predict_1 = predicted.detach().numpy().reshape(len(predicted),-1)
print(predict_1)
train = np.hstack((Y_test , predict_1))
print(train)
X=X_test.detach().numpy().reshape(len(predicted),-1)
train= np.hstack((train,X))
print(train)
np.savetxt('./all.csv',train,delimiter=',')


print(train)



