from torch import nn
from mamba_ssm import Mamba as Mamba_ssm


class Mamba(nn.Module):
    def __init__(self, input_dim, seq_len, d_model=64, d_state=16, d_conv=4, expand=2, use_fast_path=False):
        """
        Wrapper for the Mamba model for prediction tasks.

        Args:
            input_dim (int): Dimension of the input features.
            seq_len (int): Length of the input sequence.
            d_model (int): Model dimension (embedding size).
            d_state (int): SSM state expansion factor.
            d_conv (int): Local convolution width.
            expand (int): Block expansion factor.
            use_fast_path (bool): Use the fast-path implementation if True.
        """
        super(MambaPredictor, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.d_model = d_model

        # Input embedding layer
        self.input_embedding = nn.Linear(input_dim, d_model)

        # Mamba model
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=use_fast_path,
        )

        # Fully connected output layer
        self.fc = nn.Linear(d_model, 1)  # Assuming regression or binary classification

    def forward(self, x):
        """
        Forward pass for the MambaPredictor.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_dim] or [batch_size, input_dim].

        Returns:
            torch.Tensor: Predicted output.
        """
        if len(x.shape) == 2:  # If input is [batch_size, input_dim], add seq_len=1
            x = x.unsqueeze(1)  # Shape becomes [batch_size, 1, input_dim]

        # Embed input to match Mamba's `d_model`
        x = self.input_embedding(x)  # Shape becomes [batch_size, seq_len, d_model]

        # Pass through Mamba
        x = self.mamba(x)  # Shape remains [batch_size, seq_len, d_model]

        # Take the last time step's output
        x = x[:, -1, :]  # Shape becomes [batch_size, d_model]

        # Pass through the fully connected layer
        x = self.fc(x)  # Shape becomes [batch_size, 1]
        return x