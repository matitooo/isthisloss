import torch
import torch.nn as nn
import torch.nn.functional as F

class AdditiveAttention(nn.Module):
    """
    Implements additive (Bahdanau) attention.
    
    Given LSTM outputs (h_i), it computes:
      e_i = v^T * tanh(W * h_i)
      alpha_i = softmax(e_i)
      context = sum_i(alpha_i * h_i)
    
    Assumes lstm_outputs shape: (batch_size, seq_len, hidden_dim)
    """
    def __init__(self, hidden_size):
        super(AdditiveAttention, self).__init__()
        self.hidden_size = hidden_size
        
        # Project LSTM outputs from hidden_size to hidden_size.
        self.W = nn.Linear(hidden_size, hidden_size)
        # Project from hidden_size to a scalar score per token.
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_outputs):
        # lstm_outputs: (batch_size, seq_len, hidden_dim)
        score = torch.tanh(self.W(lstm_outputs))      # (B, T, H)
        score = self.v(score).squeeze(-1)             # (B, T)
        
        # Apply softmax to obtain attention weights along the time dimension.
        attn_weights = F.softmax(score, dim=1)        # (B, T)
        
        # Expand dimensions for weighted sum.
        attn_weights_expanded = attn_weights.unsqueeze(-1)   # (B, T, 1)
        context = torch.sum(attn_weights_expanded * lstm_outputs, dim=1)  # (B, H)
        
        return context, attn_weights