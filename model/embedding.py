import gensim.models
import torch
import numpy as np
import torch.nn as nn

def create_embedding_rep(tweets,embedding_size):
# Create and train the Word2Vec mode
    embedding_rep = gensim.models.Word2Vec(sentences=tweets, vector_size=embedding_size, window=5, min_count=1)
    return embedding_rep

def create_glove_embeddings():  
    #retrieve embedding data    
    data = np.load("data/glove/embeddings_dict.npz", allow_pickle=True)
    words = data["words"]
    vectors = data["vectors"]
    #create embedding dictionary
    glove_embeddings = {word: vectors[i] for i, word in enumerate(words)}
    return glove_embeddings

def tweet_embedding(embedding_model,tweet,embedding_size):
    #create empty tensor for collecting embedded words
    embedding=torch.zeros(size=(len(tweet),embedding_size))
    #loop through words
    for i in range(len(tweet)):
        #updates tensor with the word embedding
        embedded_word=embedding_model.wv[tweet[i]]
        embedding[i,:]=torch.tensor(embedded_word)
    #transpose the tensor to be used in the classifier
    embedding=embedding.T
    #return the tensor
    return embedding

    
    