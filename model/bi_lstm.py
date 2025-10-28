import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#For testing import:
import torch.nn.utils.rnn as rnn_utils
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.embedding import create_glove_embeddings
from utils import embed_tweets

# Check if GPU is available and set it as default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

class BILSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers):
        """
        A bidirectional LSTM.
        """
        super(BILSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
      
        self.lstm_layer = nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.n_layers,
            batch_first=True,
            bidirectional=True
        )

    def forward(self,inp,states):
        #LSTM LAYER
        output, (hn, cn) = self.lstm_layer(inp, states)
        return output,(hn,cn)

    def init_hidden(self, batch_size=1):
        # For bidirectional LSTM, the number of directions is 2.
        h0 = torch.zeros(self.n_layers * 2, batch_size, self.hidden_size,
                         device=self.lstm_layer.weight_ih_l0.device)
        c0 = torch.zeros(self.n_layers * 2, batch_size, self.hidden_size,
                         device=self.lstm_layer.weight_ih_l0.device)
        return h0, c0


def test_pipeline():
     # Create dummy tweets and labels.
    tweets = ["is this loss", "very sarcastic", "definitely not"]
    labels = [0, 1, 1]
    
    glove_embedder = create_glove_embeddings()
    
   
  # This function applies preprocessing, embeds tweets, and extracts manual features.
    batches, label_batches, featurized = embed_tweets(tweets, labels, glove_embedder, batch_size=2)
    print("embed_tweets produced {} batch(es).".format(len(batches)))

    #Work with the first batch.
    first_batch = batches[0]
    # Convert the PackedSequence to a padded tensor.
    # Since embed_tweets was called with batch_first=True, padded tensor shape is (batch_size, seq_len, input_size)
    padded, lengths = rnn_utils.pad_packed_sequence(first_batch, batch_first=True)
    print("Padded batch shape (batch_first=True):", padded.shape)
    
    #Print out the manual features for this batch.
    print("Manual features shape for first batch:", featurized[0].shape)

    #Instantiate the BILSTM model.
    # Here, input_size is set to 100 (the expected GloVe embedding dimension),
    # hidden_size is 256, and n_layers is 2.
    input_size = 100
    hidden_size = 256
    n_layers = 2
    model = BILSTM(input_size, hidden_size, n_layers)
    
    # Initialize the hidden state for the batch.
    batch_size = padded.size(0)
    hidden_state = model.init_hidden(batch_size=batch_size)
    
    # Run a forward pass through the BILSTM.
    output, (hn, cn) = model(padded, hidden_state)
    
    print("\n=== BILSTM Model Test ===")
    print("Input shape (batch_size, seq_len, input_size):", padded.shape)
    print("Output shape:", output.shape)       # Expected: (batch_size, seq_len, hidden_size)
    print("Hidden state shape:", hn.shape)       # Expected: (n_layers*2, batch_size, hidden_size)
    print("Cell state shape:", cn.shape)         

if __name__ == "__main__":
    test_pipeline()