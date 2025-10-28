import argparse
import wandb
from model.classifiers.bi_lstm_att_feat_classifier import BiLSTMAttentionFeaturesClassifier
from model.classifiers.lstm_att_classifier import LSTMAttentionClassifier
from model.classifiers.lstm_classifier import LSTMClassifier
from model.classifiers.lstm_feat_classifier import LSTMFeaturesClassifier
from model.classifiers.lstm_att_feat_classifier import LSTMAttentionFeaturesClassifier
from training import train_kfold_cross_validation
from testing import model_test
from utils import load_sweep_config

def main():
    #Define models dict used to represent model architectures
    models_dict={"lstm":"LSTM","lstm-features":"LSTMFeatures","lstm-attention":"LSTMAttention","lstm-attention-features":"LSTMAttentionFeatures","bi-lstm-attention-features":"BILSTMAttentionFeatures"}
    
    #Create Argument parser for mode and model
    parser = argparse.ArgumentParser(
        description='Train or tune model with different architectures'
    )
    #Define the argument model
    parser.add_argument(
        '--model', dest='model',
        choices=['bi-lstm-attention-features', 'lstm','lstm-attention','lstm-features','lstm-attention-features'],
        help='Choose one model architecture from this list {bi-lstm-attention-features, lstm,lstm-attention,lstm-features,lstm-attention-features}', type=str,
        required=True
    )
    #Define the argument mode
    parser.add_argument(
        '--mode', dest='mode',
        choices=['training','tuning','testing'],
        help='Choose one mode from this list {training,tuning,testing}',
        required=True
    )
    
    args = parser.parse_args()
    if args.mode == 'training':
      # Initialize wandb run; hyperparameters will be provided by the sweep agent.
        wandb.init(config={
        "lr": 0.0001,
        "epochs": 100,        # use a small number for testing
        "n_layers": 1,
        "h_size": 128,
        "batch_size": 32,
        
        })
        config = wandb.config
        lr = config.lr
        n_epochs = config.epochs
        n_layers = config.n_layers
        h_size = config.h_size
        batch_size = config.batch_size
        #Train model using k-fold cross validation
        train_kfold_cross_validation(args.model,h_size,n_layers,batch_size,n_epochs,lr)
              
    if args.mode=='tuning':
        #Load sweep config from yaml file
        sweep_config = load_sweep_config("sweep_config.yaml")
        #Retrieves command parameters
        sweep_config["parameters"]["architecture"]["value"]=models_dict[args.model]
        #Launch training command  
        sweep_config["command"] = ["python", "main.py", "--mode", "training", "--model", args.model]
        sweep_id = wandb.sweep(sweep_config, project="project_name")  
        wandb.agent(sweep_id)

    if args.mode=='testing':
        #Set required parameters to optimal configuration
        n_layers=1
        h_size=128
        #Launch model test
        model_test(args.model,h_size,n_layers)
        
    
              
        
if __name__ == "__main__":
    main()