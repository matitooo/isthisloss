import pandas
import numpy as np 
import torch
import torch.nn.utils.rnn as rnn_utils
from preprocess_data import preprocess_text
from features import manual_features
import yaml

def generate_train(source):
    """Retrieve  the necessary data from the train set CSV. Return a list
    of tweets and labels"""
    df=pandas.read_csv(source)
    tweets=[]
    labels=[]
    for i in range (df.shape[0]):
        tweet=df.loc[i]["tweet"]
        label=df.loc[i]["sarcastic"]
        tweets.append(tweet)
        labels.append(int(label))
    return tweets,labels


def generate_test(source):
    """Retrieve the necessary data from the test set CSV. Different from 
    the training one as the test CSV is organized differently """
    df=pandas.read_csv(source)
    tweets=[]
    labels=[]
    for i in range (df.shape[0]):
        tweet=df.loc[i]["text"]
        label=df.loc[i]["sarcastic"]
        tweets.append(tweet)
        labels.append(int(label))
    return tweets,labels

def create_glove_embeddings():  
    #retrieve embedding data    
    data = np.load("data/glove/embeddings_dict.npz", allow_pickle=True)
    words = data["words"]
    vectors = data["vectors"]
    #create embedding dictionary
    glove_embeddings = {word: vectors[i] for i, word in enumerate(words)}
    return glove_embeddings

def embed_tweets(tweets,labels,embedder,batch_size=1,random_seed=10  ):
    """
    Compute the confusion matrix for the sarcasm detector.
    The featurized_data flag is required cause feature-based models require features tensors
    as additional input.
    Parameters:
        tweets: tweets to embed:
        embedder: embedder dictionary
        batch_size: batch size for padding and packing
        labels: list of tweet labels.
        random_seed: seed for experiment reproducibility
        featurized_data: (optional) List of feature batches corresponding to each data batch.

    Returns:
        Packed and padded batches of tweets, labels and manually extracted features
    """
    #Tweets preprocessing
    processed_tweets_lists=[preprocess_text(tweets[i]).split(" ") for i in range(len(tweets))]

    filtered_tweets=[]
    filtered_labels=[]
    #Removing tweets that feature no word known to the embedder
    for i in range(len(processed_tweets_lists)):
        tweet=processed_tweets_lists[i]
        filtered_tweet=[]
        for j in range(len(tweet)):
            word=tweet[j]
            if embedder.get(word) is not None:
                filtered_tweet.append(word)
        if len(filtered_tweet)!=0:
            filtered_tweets.append(filtered_tweet)
            filtered_labels.append(labels[i])

    
    #Set random seed and shuffle data
    np.random.seed(random_seed)
    indexes=np.arange(len(filtered_tweets))
    np.random.shuffle(indexes)

    shuffled_tweets_for_qualitative=[tweets[i] for i in indexes]
    
    shuffled_tweets=[filtered_tweets[i] for i in indexes]
    shuffled_labels=[filtered_labels[i] for i in indexes]

    embedded_tweets=[] 

    #Loop through tweet and embed
    for tweet in shuffled_tweets:
        tweet_size=len(tweet)
        embedding=torch.zeros(size=(100,tweet_size))
        for i in range(len(tweet)):
            word_embedding=embedder[tweet[i]]
            embedding[:,i]=torch.from_numpy(word_embedding)
        embedded_tweets.append(embedding.T)

    #Split data in batches
    batches = [embedded_tweets[i:i + batch_size] for i in range(0, len(embedded_tweets), batch_size)]
    tweet_batches=[tweets[i:i + batch_size] for i in range(0, len(tweets), batch_size)]
    label_batches = [shuffled_labels[i:i + batch_size] for i in range(0, len(labels), batch_size)]

    packed_batches = []
    batch_labels = []
    featurized_tweets=[]

   #Pads tweets with zeros to allow batch training in LSTM-based models
    for batch, label_batch,tweet_batch in zip(batches, label_batches,tweet_batches):
        lengths = torch.tensor([t.shape[0] for t in batch])
        lengths, desc_ord = lengths.sort(descending=True)

        batch = [batch[i] for i in desc_ord]
        label_batch = [label_batch[i] for i in desc_ord]
        tweet_batch=[tweet_batch[i] for i in desc_ord]
        
        featurized_batch=manual_features(tweet_batch)

        padded_batch = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=0)

        packed_batch = rnn_utils.pack_padded_sequence(padded_batch, lengths.cpu(), batch_first=True, enforce_sorted=True)

        packed_batches.append(packed_batch)
        batch_labels.append(label_batch)
        
        featurized_tweets.append(featurized_batch)
        
    #Returns packed batches 
    return packed_batches,batch_labels,featurized_tweets

def load_sweep_config(path):
    #Load the yaml sweep from file
    with open(path, "r") as f:
        return yaml.safe_load(f)