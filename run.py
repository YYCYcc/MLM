import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
#from model.init import model
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from model.Multiagent import MultiAgentFusion
# GNNs (Graph Neural Network) 
from model.mlp import MLPModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# MAMBA 
import torch
import torch.nn as nn
#from model.mamba import Mamba
# LSTM 
from model.lstm import LSTM
# GRU 
from model.gru import GRUModel
# Neural Network 
from model.neural import NeuralODE
# Transformer 
from model.transformer import TransformerModel
from sklearn.preprocessing import LabelEncoder
import numpy as np
import shap
from mamba_ssm import Mamba
import os

torch.cuda.empty_cache()
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"
print(torch.__version__)  # 查看torch当前版本号
print(torch.version.cuda)  # 编译当前版本的torch使用的cuda版本号
print(torch.cuda.is_available())  # 查看当前cuda是否可用于当前版本的Torch
data = pd.read_csv('./TrafficLabelling/end.csv').values[0:10000]
le = LabelEncoder()
le.fit(data[:,-1])
data[:,-1]= le.transform(data[:,-1])
X_train, X_test, y_train, y_test = train_test_split(data[:,:-1].astype('float64'), data[:,-1].astype('float64'), test_size=0.2)
X_train=X_train[:, :5] 
X_test=X_test[:, :5] 
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
num_classes = len(np.unique(y_train))
class Agent:
    def __init__(self, model, optimizer, loss_fn, device):
        self.model = model.to(device)  # 把模型移动到指定设备
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.reward = 0
        self.device = device

    def train(self, X, y):
        self.model.train()
        inputs = torch.tensor(X, dtype=torch.float32).to(self.device)  # 将输入数据放到指定设备
        targets = torch.tensor(y, dtype=torch.long).to(self.device)  # 将标签放到设备
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, X):
        self.model.eval()
        inputs = torch.tensor(X, dtype=torch.float32, requires_grad=True).to(self.device)
        outputs = self.model(inputs)
        return outputs

    def update_reward(self, reward):
        self.reward += reward

    def adjust_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(0.001, 0.01 + self.reward * 0.001)

model = MLPModel(input_size=5, hidden_size=64, output_size=num_classes)
mlp_model = MLPModel(input_size=5, hidden_size=64, output_size=num_classes)
gru_model = GRUModel(input_size=5, hidden_layer_size=64, num_layers=2, output_size=num_classes)
lstm_model = LSTM(input_size=5, hidden_layer_size=64, num_layers=2, output_size=num_classes)
batch, length, dim = 2, 64, 768

mamba_model = Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=dim, # Model dimension d_model
    d_state=16,  # SSM state expansion factor # 64
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
    use_fast_path=False,
).to("cuda")
neural_model = NeuralODE()
transformer_model = TransformerModel(output_size=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建每个代理时传入设备
agents = [
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
        model=lstm_model,
        optimizer=optim.Adam(lstm_model.parameters(), lr=0.1),
        loss_fn=nn.CrossEntropyLoss(),
        device=device
    ),
    Agent(
        model=mamba_model,
        optimizer=optim.Adam(mamba_model.parameters(), lr=0.1),
        loss_fn=nn.CrossEntropyLoss(),
        device=device
    ),
    Agent(
        model=neural_model,
        optimizer=optim.Adam(neural_model.parameters(), lr=0.1),
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
for iteration in range(num_iterations):
    print(f"Iteration {iteration+1}")
    epoch_train_losses = []
    # Each agent trains on the batch data
    agent_outputs = []
    for agent in agents:
        for epoch in range(100):  # Number of epochs for each agent
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                outputs = agent.predict(inputs)
                val_loss = agent.loss_fn(outputs, targets.to(device))
                epoch_train_losses.append(val_loss.item())
                agent_outputs.append(outputs)
            agent_outputs = [agent.predict(X_train_tensor) for agent in agents]
            all_agent_outputs = torch.stack(agent_outputs, dim=0)  # Combine agent outputs

# Feedback distribution (ensure requires_grad=False for FB distribution)
            fb_distribution = all_agent_outputs.mean(dim=0).detach()
            # Feedback step
            for i, agent in enumerate(agents):
                agent_output = agent.predict(X_train_tensor)
                fb_loss = agent.loss_fn(agent_output, fb_distribution)
                agent.optimizer.zero_grad()
                fb_loss.backward()  # Ensure no reuse of computation graph
                agent.optimizer.step()

        

    # Compute Shapley values for each agent based on the feedback
    shapley_values = []
    for agent in agents:
        explainer = shap.KernelExplainer(agent.predict, X_train)
        shap_values = explainer.shap_values(X_test)
        shapley_values.append(np.mean(shap_values))
    
    # Reward and adjust learning rate based on Shapley values
    for i, agent in enumerate(agents):
        reward = shapley_values[i]
        agent.update_reward(reward)
        agent.adjust_learning_rate()

    # Display rewards after each iteration
    for i, agent in enumerate(agents):
        print(f"Agent {i+1} reward after iteration {iteration+1}: {agent.reward}")
     # 保存最好的模型
     # 记录并显示每个epoch的训练损失
    avg_train_loss = np.mean(epoch_train_losses)
    print(f"Average Training Loss for Iteration {iteration + 1}: {avg_train_loss:.4f}")
    train_losses.append(avg_train_loss)
 
    if iteration == 0 or avg_train_loss < min(train_losses):
        torch.save(agents[0].model.state_dict(), f"best_model_iteration_{iteration+1}.pth")
# Final output for each agent's accumulated reward
# 绘制训练损失曲线
plt.plot(range(num_iterations), train_losses, label="Training Loss")
plt.plot(range(num_iterations), val_losses, label="Validation Loss")
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.show()
for i, agent in enumerate(agents):
    print(f"Final Reward for Agent {i+1}: {agent.reward}")