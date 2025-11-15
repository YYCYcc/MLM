import torch
import torch.nn as nn
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.gru = nn.GRU(input_size, hidden_layer_size, num_layers=num_layers)
        self.linear = nn.Linear(hidden_layer_size, output_size)
    
    def forward(self, input_seq):
        gru_out, _ = self.gru(input_seq.view(len(input_seq), 1, -1))
        predictions = self.linear(gru_out[:, -1, :])  # Only use last time step's output
        return predictions

