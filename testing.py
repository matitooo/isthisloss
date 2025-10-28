import torch 
import numpy as np 
from utils import generate_train, generate_test, create_glove_embeddings, embed_tweets
from preprocess_data import *
import pickle
from tqdm import tqdm 
from model.classifiers.bi_lstm_att_feat_classifier import BiLSTMAttentionFeaturesClassifier
from model.classifiers.lstm_att_classifier import LSTMAttentionClassifier
from model.classifiers.lstm_classifier import LSTMClassifier
from model.classifiers.lstm_feat_classifier import LSTMFeaturesClassifier
from model.classifiers.lstm_att_feat_classifier import LSTMAttentionFeaturesClassifier
from metrics import compute_metrics, confusion
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_dict={"lstm":"LSTM","lstm-features":"LSTMFeatures","lstm-attention":"LSTMAttention","lstm-attention-features":"LSTMAttentionFeatures","bi-lstm-attention-features":"BILSTMAttentionFeatures"}

def model_test(model_name,h_size,n_layers):
    #Retrieve test and train data
    test_source = "data/test/test.csv"
    train_source = "data/train/train.csv"
    train_tweets,train_labels= generate_train(train_source)
    test_tweets, test_labels = generate_test(test_source)
    embedder = create_glove_embeddings()
    
    #Prepare train data and test data in batch size of 1 to perform the evaluation on 
    train_batches, train_label_batches, train_featurized = embed_tweets(train_tweets, train_labels, embedder, 1)
    test_batches, test_label_batches, test_featurized = embed_tweets(test_tweets, test_labels, embedder, 1)  
    
    #Check model name and retrieves weights of a trained instance from the weights folder
    if model_name=='bi-lstm-attention-features':
            model=BiLSTMAttentionFeaturesClassifier(100,h_size,n_layers,8)
            model.load_state_dict(torch.load("weights/"+model_name+"_weights.pth",map_location=device,weights_only=True))
            model.eval()
            features_flag=True
    if model_name=='lstm-attention-features':
            model=LSTMAttentionFeaturesClassifier(100,h_size,n_layers,8)
            model.load_state_dict(torch.load("weights/"+model_name+"_weights.pth",map_location=device,weights_only=True))
            model.eval()
            features_flag=True
    if model_name=='lstm-features':
            model=LSTMFeaturesClassifier(100,h_size,n_layers,8)
            model.load_state_dict(torch.load("weights/"+model_name+"_weights.pth",map_location=device,weights_only=True))
            model.eval()
            features_flag=True
    if model_name=='lstm-attention':
            model=LSTMAttentionClassifier(100,h_size,n_layers,8)
            model.load_state_dict(torch.load("weights/"+model_name+"_weights.pth",map_location=device,weights_only=True))
            model.eval()
            features_flag=False
    if model_name=='lstm':
            model=LSTMClassifier(100,h_size,n_layers,8)
            model.load_state_dict(torch.load("weights/"+model_name+"_weights.pth",map_location=device,weights_only=True))
            model.eval()
            features_flag=False
    
    #Compute metrics for the selected model
    if features_flag:
        training_metrics = compute_metrics(model, train_batches,train_label_batches, train_featurized)
        test_metrics = compute_metrics(model, test_batches,test_label_batches, test_featurized )
    else:
        training_metrics = compute_metrics(sarcastic_detector=model, packed_batches=train_batches,batch_labels=train_label_batches,features_flag=False)
        test_metrics = compute_metrics(sarcastic_detector=model, packed_batches=test_batches,batch_labels=test_label_batches,features_flag=False)
    
    #Print computed  metrics
    print("Model Architecture: %s"%(models_dict[model_name]))
    print("Scores obtained evaluating on the test set")
    print("Test Accuracy: %f"%(test_metrics["accuracy"]))
    print("Test Precision: %f"%(test_metrics["precision"]))
    print("Test Recall: %f"%(test_metrics["recall"]))
    print("Test F1 score: %f"%(test_metrics["f1"]))
    print("Scores obtained evaluating on the training set")
    print("Training Accuracy: %f"%(training_metrics["accuracy"]))
    print("Training Precision: %f"%(training_metrics["precision"]))
    print("Training Recall: %f"%(training_metrics["recall"]))
    print("Training F1 score: %f"%(training_metrics["f1"]))