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


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)


def train_kfold_cross_validation(model_name,h_size,n_layers,batch_size,n_epochs,lr):
    #Select number of fold in k-fold cross validation
    k_number = 5
 
        
            
    for fold_idx in range(k_number):
        wandb.define_metric(f"fold_{fold_idx}/epoch")
        wandb.define_metric(f"fold_{fold_idx}/train_loss", step_metric=f"fold_{fold_idx}/epoch", summary="min")
        wandb.define_metric(f"fold_{fold_idx}/train_accuracy", step_metric=f"fold_{fold_idx}/epoch", summary="max")
        wandb.define_metric(f"fold_{fold_idx}/train_precision", step_metric=f"fold_{fold_idx}/epoch", summary="max")
        wandb.define_metric(f"fold_{fold_idx}/train_recall", step_metric=f"fold_{fold_idx}/epoch", summary="max")
        wandb.define_metric(f"fold_{fold_idx}/train_f1", step_metric=f"fold_{fold_idx}/epoch", summary="max")
        wandb.define_metric(f"fold_{fold_idx}/validation_loss", step_metric=f"fold_{fold_idx}/epoch", summary="min")
        wandb.define_metric(f"fold_{fold_idx}/validation_accuracy", step_metric=f"fold_{fold_idx}/epoch", summary="max")
        wandb.define_metric(f"fold_{fold_idx}/validation_precision", step_metric=f"fold_{fold_idx}/epoch", summary="max")
        wandb.define_metric(f"fold_{fold_idx}/validation_recall", step_metric=f"fold_{fold_idx}/epoch", summary="max")
        wandb.define_metric(f"fold_{fold_idx}/validation_f1", step_metric=f"fold_{fold_idx}/epoch", summary="max")
        
    config = wandb.config
    
    # Load and prepare data
    train_source = "data/train/train.csv"
    test_source = "data/test/test.csv"
    tweets, labels = generate_train(train_source)
    test_tweets, test_labels = generate_test(test_source)
    embedder = create_glove_embeddings()
    
    #Set random seed for experiment reproducibility 
    random_seed=42
    np.random.seed(random_seed)

    #Prepare and shuffle train set
    len_train_set = len(tweets)
    step = int(len_train_set / k_number) + 1
    step_indexes = [x for x in range(0, len_train_set, step)] + [len_train_set]

    shuffled_indexes=np.arange(len_train_set)
    np.random.shuffle(shuffled_indexes)

    shuffled_tweets=[tweets[i] for i in shuffled_indexes]
    shuffled_labels=[labels[i] for i in shuffled_indexes]

    # Prepare test data
    test_batches, test_label_batches, test_featurized = embed_tweets(test_tweets, test_labels, embedder, batch_size)  
    
    # List to store validation losses for all folds
    global_val_losses = []
    
    #Loops through folds
    for fold_idx in tqdm(range(k_number)):
        
        #Create the model based on the selected architecture
        if model_name=='bi-lstm-attention-features':
            model=BiLSTMAttentionFeaturesClassifier(100,h_size,n_layers,8)
            features_flag=True
        if model_name=='lstm-attention-features':
            model=LSTMAttentionFeaturesClassifier(100,h_size,n_layers,8)
            features_flag=True
        if model_name=='lstm-features':
            model=LSTMFeaturesClassifier(100,h_size,n_layers,8)
            features_flag=True
        if model_name=='lstm-attention':
            model=LSTMAttentionClassifier(100,h_size,n_layers,8)
            features_flag=False
        if model_name=='lstm':
            model=LSTMClassifier(100,h_size,n_layers,8)
            features_flag=False
        
        
        #Prepare validation set
        validation_indexes = [j for j in range(step_indexes[fold_idx], step_indexes[fold_idx+1])]
        train_indexes = [j for j in range(len_train_set) if j not in validation_indexes]
    
        iteration_train_set = [shuffled_tweets[j] for j in train_indexes]
        iteration_train_labels = [shuffled_labels[j] for j in train_indexes]
            
        iteration_validation_set = [shuffled_tweets[j] for j in validation_indexes]
        iteration_validation_labels = [shuffled_labels[j] for j in validation_indexes]
    
        train_batches, train_label_batches, train_featurized = embed_tweets(iteration_train_set, iteration_train_labels, embedder, batch_size)
        validation_batches, validation_label_batches, validation_featurized = embed_tweets(iteration_validation_set, iteration_validation_labels, embedder, batch_size)
       
       #Create optimizer and loss function
        wandb.watch(model, log="parameters", log_freq=100)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()

        test_interval = max(1, int(n_epochs * 0.05))
        
        #Loop through epochs and perform training
        for epoch in tqdm(range(n_epochs)):
            model.train()
            epoch_loss = 0.0
            for batch, lab_batch, feat_batch in zip(train_batches, train_label_batches, train_featurized):
                optimizer.zero_grad()
                if features_flag:
                    predictions, _ = model(batch, feat_batch)
                else:
                    predictions, _ = model(batch)
                target = torch.tensor(lab_batch, dtype=torch.long, device=predictions.device)
                loss = criterion(predictions, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(lab_batch)
            total_samples = len(iteration_train_set)
            epoch_avg_loss = epoch_loss / total_samples
            wandb.log({f"fold_{fold_idx}/epoch": epoch, f"fold_{fold_idx}/train_loss": epoch_avg_loss})
            
            if epoch <101 or (epoch + 1) % test_interval == 0 or epoch == n_epochs - 1:
                if features_flag:
                    train_metrics = compute_metrics(model, train_batches, train_label_batches, train_featurized)
                    validation_metrics = compute_metrics(model, validation_batches,validation_label_batches, validation_featurized )
                else:
                    train_metrics = compute_metrics(sarcastic_detector=model, packed_batches=train_batches,batch_labels=train_label_batches,features_flag=False)
                    validation_metrics = compute_metrics(sarcastic_detector=model, packed_batches=validation_batches,batch_labels=validation_label_batches,features_flag=False)
                # Compute average validation loss for this epoch
                validation_loss = 0.0
                total_val_samples = 0
                with torch.no_grad():
                    for batch, lab_batch, feat_batch in zip(validation_batches, validation_label_batches, validation_featurized):
                        if features_flag:
                            predictions, _ = model(batch, feat_batch)
                        else:
                            predictions, _ = model(batch)
                        target = torch.tensor(lab_batch, dtype=torch.long, device=predictions.device)
                        loss = criterion(predictions, target)
                        validation_loss += loss.item() * len(lab_batch)
                        total_val_samples += len(lab_batch)
                validation_loss /= total_val_samples
                wandb.log({
                    f"fold_{fold_idx}/epoch": epoch,
                    f"fold_{fold_idx}/train_accuracy": train_metrics["accuracy"],
                    f"fold_{fold_idx}/train_precision": train_metrics["precision"],
                    f"fold_{fold_idx}/train_recall": train_metrics["recall"],
                    f"fold_{fold_idx}/train_f1": train_metrics["f1"],
                    f"fold_{fold_idx}/validation_accuracy": validation_metrics["accuracy"],
                    f"fold_{fold_idx}/validation_precision": validation_metrics["precision"],
                    f"fold_{fold_idx}/validation_recall": validation_metrics["recall"],
                    f"fold_{fold_idx}/validation_f1": validation_metrics["f1"],
                    f"fold_{fold_idx}/validation_loss": validation_loss
                })
        global_val_losses.append(validation_loss)
        
        #Compute final metrics
        if features_flag:
            final_train_metrics = compute_metrics(model, train_batches,train_label_batches, train_featurized)
            final_test_metrics = compute_metrics(model, test_batches,test_label_batches, test_featurized )
            final_val_metrics = compute_metrics(model, validation_batches, validation_label_batches,validation_featurized)
        else:
            final_train_metrics = compute_metrics(sarcastic_detector=model, packed_batches=train_batches,batch_labels=train_label_batches,features_flag=False)
            final_test_metrics = compute_metrics(sarcastic_detector=model, packed_batches=test_batches,batch_labels=test_label_batches,features_flag=False)
            final_val_metrics = compute_metrics(sarcastic_detector=model, packed_batches=validation_batches,batch_labels=validation_label_batches,features_flag=False)
    
        #Log final metrics   
        wandb.log({
            f"fold_{fold_idx}/final_train_accuracy": final_train_metrics["accuracy"],
            f"fold_{fold_idx}/final_train_precision": final_train_metrics["precision"],
            f"fold_{fold_idx}/final_train_recall": final_train_metrics["recall"],
            f"fold_{fold_idx}/final_train_f1": final_train_metrics["f1"],
            f"fold_{fold_idx}/final_test_accuracy": final_test_metrics["accuracy"],
            f"fold_{fold_idx}/final_test_precision": final_test_metrics["precision"],
            f"fold_{fold_idx}/final_test_recall": final_test_metrics["recall"],
            f"fold_{fold_idx}/final_test_f1": final_test_metrics["f1"],
            f"fold_{fold_idx}/final_validation_accuracy": final_val_metrics["accuracy"],
            f"fold_{fold_idx}/final_validation_precision": final_val_metrics["precision"],
            f"fold_{fold_idx}/final_validation_recall": final_val_metrics["recall"],
            f"fold_{fold_idx}/final_validation_f1": final_val_metrics["f1"],
            f"fold_{fold_idx}/final_validation_loss": validation_loss
        })
        
        #Compute confusion matrix
        if features_flag:
            cm = confusion(model, test_batches, test_label_batches, test_featurized)
        else:
            cm = confusion(model, test_batches, test_label_batches,featurized_data=None)
        
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        wandb.log({"confusion_matrix": wandb.Image(plt)})
        
        #Computes predictions
        all_train_preds = []
        for batch, lab_batch, feat_batch in zip(train_batches, train_label_batches, train_featurized):
            with torch.no_grad():
                if features_flag:
                    outputs, _ = model(batch, feat_batch)
                else:
                    outputs, _ = model(batch)
                all_train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy().tolist())
        
        all_test_preds = []
        for batch, lab_batch, feat_batch in zip(test_batches, test_label_batches, test_featurized):
            with torch.no_grad():
                if features_flag:
                    outputs, _ = model(batch, feat_batch)
                else:
                    outputs, _ = model(batch)
                all_test_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy().tolist())
        
        # Save predictions as pickle files
        with open("train_predictions.pkl", "wb") as f:
            pickle.dump(all_train_preds, f)
        with open("test_predictions.pkl", "wb") as f:
            pickle.dump(all_test_preds, f)

        # Log predictions as a wandb artifact (each run gets a unique identifier)
        artifact = wandb.Artifact(f"predictions_{wandb.run.id}", type="predictions")
        artifact.add_file("train_predictions.pkl")
        artifact.add_file("test_predictions.pkl")
        wandb.log_artifact(artifact)
    
    torch.save(model.state_dict(), model_name+"_weights.pth")
    #here we make the average of the losses and the variance
    final_average_val_loss = np.mean(global_val_losses)
    final_val_loss_variance = np.var(global_val_losses)
    wandb.log({
        "final_average_val_loss": final_average_val_loss,
        "final_validation_loss_variance": final_val_loss_variance
    })
    
    wandb.finish()
    


