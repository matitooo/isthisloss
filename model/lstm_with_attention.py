import torch
import torch.nn as nn
from .attention import AdditiveAttention  # Ensure attention.py is in the same directory
import torch.nn.utils.rnn as rnn_utils

class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        """
        Deep Learning Stream: Processes text embeddings through an LSTM followed by an attention mechanism.
        
        :param input_size: Dimensionality of text embeddings.
        :param hidden_size: Hidden size for the LSTM.
        :param n_layers: Number of LSTM layers.
        """
        super(LSTMWithAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        # LSTM with batch_first=True expects input shape (batch_size, seq_len, input_size)
        self.lstm_layer = nn.LSTM(self.input_size, self.hidden_size, 1, batch_first=True)
        self.lstm_layer2 = nn.LSTM(self.hidden_size, int(self.hidden_size/2) , 1, batch_first=True)

        self.attention = AdditiveAttention(int(self.hidden_size/2))

    def forward(self, inp, states=None):
        """
        :param inp: Tensor of shape (batch_size, seq_len, input_size)
        :param states: Optional initial LSTM states.
        :return: 
            context: (batch_size, hidden_size) from attention mechanism.
            attn_weights: (batch_size, seq_len)
        """
        if states is None:
            if isinstance(inp, torch.nn.utils.rnn.PackedSequence):
                batch_size = inp.batch_sizes[0]
            else:
                batch_size = inp.size(0)
            states = self.init_hidden(batch_size=batch_size)
        output, (hn, cn) = self.lstm_layer(inp, states)  # output: (B, T, H)
        output, (hn, cn) = self.lstm_layer2(output)  # output: (B, T, H)
        padded_output, _ = rnn_utils.pad_packed_sequence(output, batch_first=True)
        context, attn_weights = self.attention(padded_output)     # context: (B, H)
        return context, attn_weights

    def init_hidden(self, batch_size=1):
        device = next(self.parameters()).device
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
        return (h0, c0)

