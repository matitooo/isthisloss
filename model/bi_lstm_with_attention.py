import torch
import torch.nn as nn
import torch.nn.functional as F
from .bi_lstm import BILSTM  
from .attention import AdditiveAttention
import torch.nn.utils.rnn as rnn_utils

class BiLSTMWithAttention(BILSTM):
    def __init__(self, input_size, hidden_size, n_layers=1):
        """
        Inherits from BILSTM (which already uses bidirectional=True)
        and adds an attention mechanism.
        
        Args:
            input_size (int): Input embedding dimension.
            hidden_size (int): LSTM hidden size per direction.
            n_layers (int): Number of LSTM layers.
        """
        super(BiLSTMWithAttention, self).__init__(input_size, hidden_size, n_layers)
        # For a bidirectional LSTM, output dimension is hidden_size * 2.
        self.attention = AdditiveAttention(hidden_size * 2)

    def forward(self, inp, states=None):
        """
        Forward pass: Get LSTM outputs from the parent BILSTM, then apply attention.
        
        Args:
            inp: Input tensor of shape (batch_size, seq_len, input_size).
            states: Optional initial LSTM states.
        
        Returns:
            context: Attention-weighted context vector (batch_size, hidden_size * 2).
            attn_weights: Attention weights over the sequence (batch_size, seq_len).
        """
        # If no initial states provided, initialize them.
        if states is None:
            if isinstance(inp, torch.nn.utils.rnn.PackedSequence):
                batch_size = inp.batch_sizes[0]
            else:
                batch_size = inp.size(0)
            states = self.init_hidden(batch_size=batch_size)
        # Call the parent BILSTM forward method.
        output, (hn, cn) = super(BiLSTMWithAttention, self).forward(inp, states)
        
        # Apply attention on the LSTM outputs.
        padded_output, _ = rnn_utils.pad_packed_sequence(output, batch_first=True)
        context, attn_weights = self.attention(padded_output)     # context: (B, H)

        return context, attn_weights

# Example usage for testing:
if __name__ == "__main__":
    # Test parameters.
    batch_size = 4
    seq_len = 10
    input_size = 100  #  GloVe embedding dimension
    hidden_size = 256
    n_layers = 2

    # Create a dummy input tensor: shape (batch_size, seq_len, input_size)
    dummy_input = torch.randn(batch_size, seq_len, input_size)
    
    # Initialize the BiLSTMWithAttention model.
    model = BiLSTMWithAttention(input_size, hidden_size, n_layers)
    # Forward pass.
    context, attn_weights = model(dummy_input)
    print("Context shape:", context.shape)            # Expected: (batch_size, hidden_size * 2)
    print("Attention weights shape:", attn_weights.shape)  # Expected: (batch_size, seq_len)