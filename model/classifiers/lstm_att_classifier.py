import torch
import torch.nn as nn
import torch.nn.functional as F
from ..lstm_with_attention import LSTMWithAttention

#This is a sarcasm detector model that uses only the deep learning stream with LSTM and attention, without any manual features.

class LSTMAttentionClassifier(nn.Module):
    def __init__(self, embedded_stack_size, lstm_hidden_size, lstm_n_layers, num_classes=2):
        """
        Classifier using LSTM with attention without manual features.
        
        Args:
            embedded_stack_size (int): Dimension of text embeddings.
            lstm_hidden_size (int): LSTM hidden size per direction.
            lstm_n_layers (int): Number of LSTM layers.
            num_classes (int): Number of output classes.
        """
        super(LSTMAttentionClassifier, self).__init__()
        self.lstm_with_attention = LSTMWithAttention(embedded_stack_size, lstm_hidden_size, lstm_n_layers)
        # The context vector from LSTMWithAttention is of size (lstm_hidden_size * 2)
        self.classifier = nn.Linear(int(lstm_hidden_size/2), num_classes)
    
    def forward(self, embedded_stack, lstm_states=None):
        """
        Forward pass.
        
        Args:
            embedded_stack: Tensor of shape (batch_size, seq_len, embedded_stack_size)
            lstm_states: Optional initial LSTM states.
            
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
            attn_weights: Tensor of shape (batch_size, seq_len)
        """
        # Process text via LSTM with attention to get a context vector
        context, attn_weights = self.lstm_with_attention(embedded_stack, lstm_states)
        # Final classification layer uses only the context vector
        logits = self.classifier(context)
        return logits, attn_weights
    
    def predict(self, embedded_stack, lstm_states=None):
        """
        Performs a forward pass, applies softmax, and returns predicted class indices.
        
        Args:
            embedded_stack: Tensor of shape (batch_size, seq_len, embedded_stack_size)
            lstm_states: Optional initial LSTM states.
            
        Returns:
            predictions: Tensor of shape (batch_size) with predicted class indices.
        """
        logits, _ = self.forward(embedded_stack, lstm_states)
        probs = F.softmax(logits, dim=-1)
        
        return torch.argmax(probs, dim=1)

# Example usage:
if __name__ == "__main__":
    # Test parameters.
    batch_size = 4
    seq_len = 10
    embedded_stack_size = 100   # e.g., GloVe embedding dimension.
    lstm_hidden_size = 256
    lstm_n_layers = 2
    num_classes = 2

    # Create dummy input tensor: shape (batch_size, seq_len, embedded_stack_size)
    dummy_input = torch.randn(batch_size, seq_len, embedded_stack_size)
    
    # Initialize the classifier.
    model = LSTMAttentionClassifier(embedded_stack_size, lstm_hidden_size, lstm_n_layers, num_classes)
    
    # Forward pass.
    logits, attn_weights = model(dummy_input)
    print("Logits shape:", logits.shape)             # Expected: (batch_size, num_classes)
    print("Attention weights shape:", attn_weights.shape)  # Expected: (batch_size, seq_len)
    
    # Prediction.
    predictions = model.predict(dummy_input)
    print("Predictions:", predictions)