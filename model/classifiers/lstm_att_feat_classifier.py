import torch
import torch.nn as nn
from ..lstm_with_attention import LSTMWithAttention
import torch.nn.utils.rnn as rnn_utils

class LSTMAttentionFeaturesClassifier(nn.Module):
    def __init__(self, embedded_stack_size, lstm_hidden_size, lstm_n_layers,n_features):
        """
        Sarcasm Detection Model using only the deep learning stream with LSTM and attention,
        without any manual features.
        
        :param embedded_stack_size: Dimension of text embeddings.
        :param lstm_hidden_size: Hidden size for the LSTM.
        :param lstm_n_layers: Number of LSTM layers.
        :param num_classes: Number of classification classes (e.g., 2 for sarcastic vs. non-sarcastic).
        """
        super(LSTMAttentionFeaturesClassifier, self).__init__()
        # Deep learning stream for text embeddings

        self.lstm_with_attention = LSTMWithAttention(embedded_stack_size, lstm_hidden_size, lstm_n_layers)
        second_hidden_size=int(lstm_hidden_size/2)+n_features
        output_size=int(second_hidden_size/2)

        # Final classifier: only uses the context vector from LSTM (no concatenation)
        self.classifier = nn.Linear(second_hidden_size,2)
        


    def forward(self, embedded_stack,extracted_features, lstm_states=None):
        """
        :param embedded_stack: Stack of tensors of shape (batch_size, seq_len, embedded_stack_size)
        :extracted features: Stack of tensors of extracted features of shape (batch_size,17)
        :param lstm_states: Optional initial LSTM states.
        :return: 
            logits: Tensor of shape (batch_size, num_classes)
            attn_weights: Attention weights from the text stream (batch_size, seq_len)
        """
        # Process text via LSTM with attention to get a context vector
        context, attn_weights = self.lstm_with_attention(embedded_stack, lstm_states)  # context: (B, lstm_hidden_size)
        # Concatenate attention output and manually extracted features
        concatenation=torch.cat((context,extracted_features),dim=-1)
        # Final classification layer uses the concatenated vector
        logits = self.classifier(concatenation)  # (B, num_classes)
        return logits, attn_weights
    
    def predict(self, embedded_stack,extracted_features):
        """
        :param embedded_stack: Tensor of shape (batch_size, seq_len, embedded_stack_size)
        :return: 
            predictions: Tensor of shape (batch_size) with 1 (sarcastic) or 0 (non-sarcastic) predictions.
        """
        logits, _ = self(embedded_stack,extracted_features)
        return torch.argmax(logits, dim=1)