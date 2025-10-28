import torch
import torch.nn as nn
import torch.nn.functional as F
from ..bi_lstm_with_attention import BiLSTMWithAttention

#This sarcasm detector model uses a bidirectional LSTM with attention and manual features.
class BiLSTMAttentionFeaturesClassifier(nn.Module):
    def __init__(self, embedded_stack_size, lstm_hidden_size, lstm_n_layers, n_features, num_classes=2):
        """
        A classifier that uses a bidirectional LSTM with attention to extract a context vector from text,
        concatenates it with manually extracted features, and then applies a classification head.
        
        Args:
            embedded_stack_size (int): Dimensionality of text embeddings.
            lstm_hidden_size (int): Hidden size of the LSTM per direction.
            lstm_n_layers (int): Number of LSTM layers.
            n_features (int): Number of manual features.
            num_classes (int): Number of target classes.
        """
        super(BiLSTMAttentionFeaturesClassifier, self).__init__()
        self.bi_lstm_with_attention = BiLSTMWithAttention(embedded_stack_size, lstm_hidden_size, lstm_n_layers)
        # For a bidirectional LSTM, the context vector from attention is expected to be of size lstm_hidden_size * 2.
        combined_dim = (lstm_hidden_size * 2) + n_features
        self.classifier = nn.Linear(combined_dim, num_classes)
        
    def forward(self, embedded_stack, extracted_features, lstm_states=None):
        """
        Forward pass.
        
        Args:
            embedded_stack: Tensor of shape (batch_size, seq_len, embedded_stack_size) – text embeddings.
            extracted_features: Tensor of shape (batch_size, n_features) – manually extracted features.
            lstm_states: Optional initial states for the LSTM.
        
        Returns:
            logits: Tensor of shape (batch_size, num_classes).
            attn_weights: Tensor of shape (batch_size, seq_len) with the attention weights.
        """
        # Process the text with the BiLSTM with attention to get a context vector.
        context, attn_weights = self.bi_lstm_with_attention(embedded_stack, lstm_states)
        # context is assumed to have shape (batch_size, lstm_hidden_size * 2).
        # Concatenate the context vector with the manual features.
        concatenated = torch.cat((context, extracted_features), dim=-1)
        # Final classification layer uses the concatenated vector
        logits = self.classifier(concatenated)
        return logits, attn_weights
    
    def predict(self, embedded_stack, extracted_features, lstm_states=None):
        """
        :param embedded_stack: Tensor of shape (batch_size, seq_len, embedded_stack_size)
        :param extracted_features: Tensor of shape (batch_size, n_features)
        :param lstm_states: Optional initial LSTM states.
        :return: 
            predictions: Tensor of shape (batch_size) with predicted class indices (e.g., 1 for sarcastic, 0 for non-sarcastic)
        """
        logits, _ = self.forward(embedded_stack, extracted_features, lstm_states)
        probs = F.softmax(logits, dim=-1)
        return torch.argmax(probs, dim=1)

# Example usage:
if __name__ == "__main__":
    batch_size = 4
    seq_len = 10
    embedded_stack_size = 100   # e.g., GloVe embedding dimension.
    lstm_hidden_size = 256
    lstm_n_layers = 2
    n_features = 17             # Number of manual features.
    
    # Create dummy inputs.
    embedded_stack = torch.randn(batch_size, seq_len, embedded_stack_size)
    extracted_features = torch.randn(batch_size, n_features)
    
    # Initialize the classifier.
    model = BiLSTMAttentionFeaturesClassifier(embedded_stack_size, lstm_hidden_size, lstm_n_layers, n_features)
    
    # Forward pass.
    logits, attn_weights = model(embedded_stack, extracted_features)
    print("Logits shape:", logits.shape)             # Expected: (batch_size, 2)
    print("Attention weights shape:", attn_weights.shape)  # Expected: (batch_size, seq_len)
    
    # Prediction.
    predictions = model.predict(embedded_stack, extracted_features)
    print("Predictions:", predictions)