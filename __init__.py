'''import torch
import torch.nn as nn
from model.Multiagent import MultiAgentFusion
# GNNs (Graph Neural Network) 
from model.mlp import GCNModel
# MAMBA 
import torch
import torch.nn as nn
from mamba import Mamba
# LSTM 
from model.lstm import LSTM
# GRU 
from model.gru import GRUModel
# Neural Network 
from model.neural import NeuralODE
# Transformer 
from model.transformer import TransformerModel

class model :
    def _init_(self):
        gnn_model = GCNModel(input_size=32, hidden_size=64, output_size=1)
        gru_model = GRUModel(input_size=1, hidden_layer_size=128, num_layers=2, output_size=1)
        lstm_model = LSTM(input_size=1, hidden_layer_size=128, num_layers=2, output_size=1)
        mamba_model = Mamba(d_model = 128, d_state = 64, d_conv = 8, expand = 2, dropout = 0.1)
        neural_model = NeuralODE(input_size=32, hidden_size=128, output_size=1)
        transformer_model = TransformerModel(output_size=1)
        models = [gnn_model, gru_model, lstm_model, mamba_model, neural_model, transformer_model]
        # 创建多代理融合模型
        model = MultiAgentFusion(models=models)'''