import torch
import torch.nn as nn

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
        print(f"Input shape before unsqueeze: {input_seq.shape}")  # Debug: Check input shape
        if len(input_seq.shape) == 2:
            input_seq = input_seq.unsqueeze(1)
            print(f"Input shape after unsqueeze: {input_seq.shape}")  # Debug: Should now be 3D

        lstm_out, _ = self.lstm(input_seq)
        print(f"LSTM output shape: {lstm_out.shape}")  # Debug: [batch_size, seq_len, hidden_layer_size]
        lstm_out_last = lstm_out[:, -1, :]
        predictions = self.linear(lstm_out_last)
        return predictions