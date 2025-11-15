from torchdiffeq import odeint

import torch
import torch.nn as nn
class NeuralODE(nn.Module):
    def __init__(self, hidden_size=128):
        super(NeuralODE, self).__init__()
        self.fc = nn.Linear(1, hidden_size)

    def forward(self, t, x):
        # 输入的 x 可能需要在这里处理
        return self.fc(x)  # 这里是简单的全连接层，也可以根据需要进行其他处理

    def solve_ode(self, x0, t):
        """
        使用 odeint 求解微分方程，传入初始条件 x0 和时间步 t。
        x0 是初始状态，t 是时间步序列。
        """
        # odeint 期望的数据格式是：x0 的形状是 [batch_size, 1]，t 是时间步序列
        return odeint(self, x0, t)

# 训练过程中调用修改后的 NeuralODE 类
def train(model, data, target, optimizer, loss_fn, device):
    model.train()
    
    # 假设 t 是一个从 0 到 1 的线性时间序列，长度是 100
    t = torch.linspace(0, 1, steps=100).to(device)  # 时间步，假设是100个时间步
    
    # 将数据转为适合的形状，假设数据是 [batch_size, 1]
    x0 = data.unsqueeze(-1).to(device)  # 增加一个维度，使其变为 [batch_size, 1]

    # 使用 solve_ode 方法求解微分方程
    outputs = model.solve_ode(x0, t)  # 这个返回的是时间步序列的输出，形状是 [batch_size, steps, hidden_size]

    # 选择最后一个时间步的输出（如果需要）
    final_output = outputs[:, -1, :]  # 取最后一个时间步的输出

    # 计算损失
    loss = loss_fn(final_output, target)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()
