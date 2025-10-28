import torch
from tqdm import tqdm 
from sklearn.metrics import confusion_matrix


def compute_metrics(sarcastic_detector, packed_batches, batch_labels, featurized_batches=None,features_flag=True):
    """
    Compute accuracy, precision, recall, and F1 score in one pass to save computation.
    The featurized flag is required as feature-based models require manual extracted features
    as additional input.
    """
    correct = 0
    total = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    if features_flag:
        for packed_batch, labels, featurized_batch in zip(packed_batches, batch_labels, featurized_batches):
            predictions = sarcastic_detector.predict(packed_batch, featurized_batch)
            labels_tensor = torch.tensor(labels, device=predictions.device)
            
            correct += (predictions == labels_tensor).sum().item()
            total += len(labels)
            tp += ((predictions == 1) & (labels_tensor == 1)).sum().item()
            tn += ((predictions == 0) & (labels_tensor == 0)).sum().item()
            fp += ((predictions == 1) & (labels_tensor == 0)).sum().item()
            fn += ((predictions == 0) & (labels_tensor == 1)).sum().item()
    
    else:
        for packed_batch, labels in zip(packed_batches, batch_labels):
            predictions = sarcastic_detector.predict(packed_batch)
            labels_tensor = torch.tensor(labels, device=predictions.device)
            
            correct += (predictions == labels_tensor).sum().item()
            total += len(labels)
            tp += ((predictions == 1) & (labels_tensor == 1)).sum().item()
            tn += ((predictions == 0) & (labels_tensor == 0)).sum().item()
            fp += ((predictions == 1) & (labels_tensor == 0)).sum().item()
            fn += ((predictions == 0) & (labels_tensor == 1)).sum().item()
    accuracy_val = correct / total
    
    precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_val = (2 * precision_val * recall_val) / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0
    
    return {
        "accuracy": accuracy_val,
        "precision": precision_val,
        "recall": recall_val,
        "f1": f1_val
    }


def confusion(sarcastic_detector, data, labels, featurized_data=None):
    """
    Compute the confusion matrix for the sarcasm detector.
    The featurized_data flag is required cause feature-based models require features tensors
    as additional input.
    Parameters:
        sarcastic_detector: The model; should support either:
          - single-sample mode: model(sample)
          - batch mode: model(batch, feat_batch)
        data: List of inputs.
        labels: List of true labels (or list of lists if in batch mode).
        featurized_data: (optional) List of feature batches corresponding to each data batch.

    Returns:
        cm: The confusion matrix computed from true labels and predictions.
    """
    predictions = []
    true_labels = []
    
    if featurized_data is not None:
        for batch, lab_batch, feat_batch in zip(data, labels, featurized_data):
            with torch.no_grad():
                outputs, _ = sarcastic_detector(batch, feat_batch)
                preds = torch.argmax(outputs, dim=1)
                predictions.extend(preds.cpu().numpy().tolist())
            true_labels.extend(lab_batch)
    else:
        for batch, lab_batch in zip(data, labels):
            with torch.no_grad():
                outputs, _ = sarcastic_detector(batch)
                preds = torch.argmax(outputs, dim=1)
                predictions.extend(preds.cpu().numpy().tolist())
            true_labels.extend(lab_batch)
    
    cm = confusion_matrix(true_labels, predictions)
    return cm