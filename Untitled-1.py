# %%

import os
print("Current working directory:", os.getcwd())
import importlib

import sys
sys.path.append('/data')
print(os.listdir('/data'))
print(os.getcwd())
from torchdiffeq import odeint
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
#from model.init import model
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from model.Multiagent import MultiAgentFusion
# GNNs (Graph Neural Network) 
from model.mlp import MLPModel
# MAMBA 
import torch
torch.cuda.empty_cache()
import torch.nn as nn
#from model.mamba import Mamba
# LSTM 
#from model.lstm import LSTM
# GRU 
from sklearn.preprocessing import StandardScaler
from model.gru import GRUModel
# Neural Network 
#from model.neural import NeuralODE
# Transformer 
from model.transformer import TransformerModel
import numpy as np
import shap
from mamba_ssm import Mamba
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"
print(torch.__version__)  # 查看torch当前版本号
print(torch.version.cuda)  # 编译当前版本的torch使用的cuda版本号
print(torch.cuda.is_available())  # 查看当前cuda是否可用于当前版本的Torch
from torch.utils.data import DataLoader, Dataset

# %%
# 定义一个 Dataset 子类，负责加载数据
class LargeDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# %%
data = pd.read_csv('/data/TrafficLabelling/end.csv').values
le = LabelEncoder()
le.fit(data[:,-1])

data[:,-1]= le.transform(data[:,-1])
num_classes = len(np.unique(data[:,-1]))


# %%
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)

        # Fully connected layer
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        #print(f"Input shape before unsqueeze: {input_seq.shape}")  # Debug: Check input shape
        if len(input_seq.shape) == 2:
            input_seq = input_seq.unsqueeze(1)
            #print(f"Input shape after unsqueeze: {input_seq.shape}")  # Debug: Should now be 3D

        lstm_out, _ = self.lstm(input_seq)
        #print(f"LSTM output shape: {lstm_out.shape}")  # Debug: [batch_size, seq_len, hidden_layer_size]
        lstm_out_last = lstm_out[:, -1, :]
        predictions = self.linear(lstm_out_last)
        return predictions

# %%
class MambaPredictor(nn.Module):
    def __init__(self, input_dim, seq_len, n_classes, d_model=16, d_state=2, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_classes = n_classes

        self.input_embedding = nn.Linear(input_dim, d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, use_fast_path=False)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        x = self.input_embedding(x)
        #print(f"After input_embedding: {x}")  # Debugging step
        x = self.mamba(x)
        #print(f"After Mamba: {x}")  # Debugging step
        x = x[:, -1, :]  # Take last time step
        x = self.fc(x)
        return x


# %%


X_train, X_test, y_train, y_test = train_test_split(data[:,:-1].astype('float64'), data[:,-1].astype('float64'), test_size=0.2)
class_0_indices = np.where(y_train == 0)[0]
other_classes_indices = np.where(y_train != 0)[0]

# Randomly select half of the '0' class samples
reduced_class_0_indices = np.random.choice(class_0_indices, size=len(class_0_indices) // 4, replace=False)

# Combine reduced '0' class indices with all other class indices
balanced_indices = np.concatenate([reduced_class_0_indices, other_classes_indices])
X_train[np.isinf(X_train)] = 0
X_train[np.isnan(X_train)] = 0
X_test[np.isinf(X_test)] = 0
X_test[np.isnan(X_test)] = 0


# %%

# Create the new balanced dataset
X_train= X_train[balanced_indices]
y_train= y_train[balanced_indices]
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train=X_train[:, :5] 
X_test=X_test[:, :5] 
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)


# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
num_classes = len(np.unique(y_train))
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)

plt.hist(y_train, bins=np.arange(num_classes + 1) - 0.5, edgecolor='black')
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.show()



# %%
class NeuralODE(nn.Module):
    def __init__(self, feature_size, n_classes, hidden_size):
        super(NeuralODE, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes

        # 假设我们需要将feature_size映射到hidden_size
        self.fc1 = nn.Linear(self.feature_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.n_classes)

    def forward(self, t, x):
        x = torch.relu(self.fc1(x))  # 第一步，映射到hidden_size
        x = self.fc2(x)  # 第二步，输出n_classes维度
        return x

    def solve_ode(self, x0, t):
        # x0 应该是[batch_size, feature_size]
        assert x0.shape[1] == self.feature_size, f"Expected feature size {self.feature_size}, got {x0.shape[1]}"
        
        # 求解ODE
        solution = odeint(self, x0, t)
        return solution


# %% [markdown]
# 

# %%
class Agent:
    def __init__(self, model, optimizer, loss_fn, device):
        """
        初始化Agent。

        参数：
        - model: 该Agent的模型
        - optimizer: 优化器
        - loss_fn: 损失函数
        - device: 设备（'cpu' 或 'cuda'）
        """
        self.model = model # 将模型移动到指定设备
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.reward = 0.0  # 初始化奖励值
        self.device = device
        self.weight = 1.0  # 初始化权重
     

    def train_NeuralODE(self, data, target):
        x0 = data  # 初始输入
        print(f"x0 shape: {x0.shape}")  # 应该是 [batch_size, feature_size]，例如 [8, 5]

        t = torch.linspace(0, 1, steps=8)  # 假设我们使用15个时间步
        print(f"t shape: {t.shape}")  # 应该是 [15]

        # 确保 x0 和 t 的形状兼容
        assert x0.shape[0] == target.shape[0], f"Batch size of x0 ({x0.shape[0]}) and target ({target.shape[0]}) must match."
        
        # 求解ODE
        outputs = self.model.solve_ode(x0, t)

        print(f"ODE output shape: {outputs.shape}")

        # 选择最后一个时间步的输出进行损失计算
        final_output = outputs[-1, :, :]  # 选择最后一个时间步的输出 [batch_size, feature_size]
        print(f"Final output shape: {final_output.shape}")

        # 对 final_output 进行最后的分类层映射
        final_output = self.model.output_layer(final_output)  # 形状变为 [batch_size, n_classes]

        # 计算损失
        loss = self.loss_fn(final_output, target)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def train(self, train_loader):
        """
        使用训练数据对模型进行训练。

        参数：
        - train_loader: DataLoader对象，用于批量加载训练数据

        返回：
        - 平均训练损失（一个epoch的平均损失）
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0  # 统计样本数量，避免数据不均衡影响损失计算

        for inputs, targets in train_loader:
            inputs, targets = inputs, targets # 将数据移动到指定设备
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)  # 模型前向传播
            
            # 检查输出和目标的形状是否一致
            assert outputs.shape == targets.shape, f"输出维度 {outputs.shape} 和目标维度 {targets.shape} 不匹配！"
            
            loss = self.loss_fn(outputs, targets)  # 计算损失
            loss.backward()  # 反向传播
            self.optimizer.step()  # 更新参数
            
            # 累加损失和样本数
            total_loss += loss.item() * len(inputs)
            total_samples += len(inputs)

        # 返回平均损失
        return total_loss / total_samples

    def predict(self, X):
        """
        使用模型进行预测。

        参数：
        - X: 输入数据，可以是numpy数组或torch.Tensor

        返回：
        - 模型的预测输出
        """
        self.model.eval()
        with torch.no_grad():  # 禁用梯度计算
            if isinstance(X, np.ndarray):  # 如果输入是numpy数组
                X = torch.tensor(X, dtype=torch.float32).to(self.device)
            elif not isinstance(X, torch.Tensor):  # 如果输入不是torch.Tensor
                raise TypeError("输入数据必须是numpy数组或torch.Tensor")
            
            outputs = self.model(X)  # 模型前向传播
        return outputs.cpu().numpy()  # 返回numpy格式的预测结果

    def update_reward(self, reward):
        """
        更新Agent的奖励值。

        参数：
        - reward: 新的奖励值
        """
        self.reward += reward

    def adjust_learning_rate(self, min_lr=0.001, max_lr=0.01):
        """
        根据奖励动态调整学习率。

        参数：
        - min_lr: 最小学习率
        - max_lr: 最大学习率
        """
        for param_group in self.optimizer.param_groups:
            new_lr = max(min_lr, min(max_lr, param_group['lr'] + self.reward * 0.001))
            param_group['lr'] = new_lr

    def adjust_weight(self):
        """
        根据奖励值调整Agent的权重。
        """
        self.weight = max(0.1, self.reward)  # 确保最小权重为0.1

    @staticmethod
    def normalize_weights(agents):
        """
        归一化所有Agent的权重，使得权重之和为1。

        参数：
        - agents: 所有Agent的列表
        """
        total_reward = sum(agent.reward for agent in agents)
        if total_reward > 0:
            for agent in agents:
                agent.weight = agent.reward / total_reward
        else:
            for agent in agents:
                agent.weight = 1.0 / len(agents)  # 如果没有奖励，均分权重


# %%

#model = MultiAgentFusion(input_size=5, hidden_size=64, output_size=num_classes)
mlp_model = MLPModel(input_size=5, hidden_size=64, output_size=num_classes)
gru_model = GRUModel(input_size=5, hidden_layer_size=64, num_layers=2, output_size=num_classes)
lstm_model = LSTM(input_size=5, hidden_layer_size=64, num_layers=2, output_size=num_classes)
batch, length, dim = 2, 64, 768
mamba_model = MambaPredictor(input_dim=5, seq_len=1, n_classes=num_classes)
neural_model = NeuralODE(feature_size=5,n_classes=num_classes, hidden_size=8)
transformer_model = TransformerModel(output_size=num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  

# %%

# 创建每个代理时传入设备
agents = [
  
    Agent(
        model=neural_model,
        optimizer=optim.Adam(neural_model.parameters(), lr=0.1),
        loss_fn=nn.CrossEntropyLoss(),
        device=device
    ),
    Agent(
        model=lstm_model,
        optimizer=optim.Adam(lstm_model.parameters(), lr=0.1),
        loss_fn=nn.CrossEntropyLoss(),
        device=device
    ),
    Agent(
        model=mlp_model,
        optimizer=optim.Adam(mlp_model.parameters(), lr=0.1),
        loss_fn=nn.CrossEntropyLoss(),
        device=device
    ),
    Agent(
        model=gru_model,
        optimizer=optim.Adam(gru_model.parameters(), lr=0.1),
        loss_fn=nn.CrossEntropyLoss(),
        device=device
    ),
  
  
    Agent(
        model=transformer_model,
        optimizer=optim.Adam(transformer_model.parameters(), lr=0.1),
        loss_fn=nn.CrossEntropyLoss(),
        device=device
    ),
]

# Main loop for training, fusion, feedback, and optimization
num_iterations = 50
train_losses = []
val_losses = []
accuracy_scores = []
agent_outputs = []

# %%
print(num_classes)

# %%
for iteration in range(num_iterations):
    print(f"Iteration {iteration + 1}")
    # 记录loss
    losses = []
    agent_outputs = []
    for i, agent in enumerate(agents):
        print(f"Training Agent {i + 1}")
        agent_loss = []
        for batch_idx, (data, target) in enumerate(train_loader):

            # Agent Training
            if i==0:
                loss = agent.train_NeuralODE(data, target)
            else:
                loss = agent.train_batch(data, target)  # 改为单批次训练
            agent_loss.append(loss)  # 将每个batch的loss存入列表
            if batch_idx % 100 == 0:  # 每100个batch打印一次
                print(f"Batch {batch_idx}, Loss: {loss}")
         # 将当前agent的所有loss加入到全局loss记录中
        losses.append(np.mean(agent_loss))  # 记录当前agent每个epoch的loss平均值
        # Generate predictions for FB distribution
        agent_outputs.append(agent.predict(X_test))
        # 将当前agent的所有loss加入到全局loss记录中
        losses.append(np.mean(agent_loss))  # 记录当前agent每个epoch的loss平均值
    # Calculate FB distribution based on agent outputs
    fb_distribution = np.mean(agent_outputs, axis=0)

    # Further training based on FB distribution feedback
    for i, agent in enumerate(agents):
        fb_loss = agent.loss_fn(
            torch.tensor(agent_outputs[i], dtype=torch.float32).to(device),
            torch.tensor(fb_distribution, dtype=torch.float32).to(device)
        )
        agent.optimizer.zero_grad()
        fb_loss.backward()
        agent.optimizer.step()

    # Shapley value calculation
    shapley_values = []
    for i, agent in enumerate(agents):
        # Randomly sample from X_train for Shapley calculation
        sample_idx = np.random.choice(len(X_train), size=100, replace=False)
        X_sample = X_train[sample_idx]
        explainer = shap.KernelExplainer(agent.predict, X_sample)
        shap_values = explainer.shap_values(X_test)
        shapley_values.append(np.mean(shap_values))

    # Reward and learning rate adjustment
    total_shapley = sum(shapley_values)
    for i, agent in enumerate(agents):
        reward = shapley_values[i] / total_shapley if total_shapley > 0 else 0
        agent.update_reward(reward)
        agent.adjust_weight()
        agent.adjust_learning_rate()

    # Display rewards
    for i, agent in enumerate(agents):
        print(f"Agent {i + 1} reward after iteration {iteration + 1}: {agent.reward}")

# %% [markdown]
# 

# %% [markdown]
# 

# %%
plt.figure(figsize=(10, 6))
for agent_name, losses in loss_curves.items():
    plt.plot(range(1, num_iterations + 1), losses, label=agent_name)

plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss Curve for Each Agent')
plt.legend()
plt.grid()
plt.show()

# %%
nvcc --version  

# %%



