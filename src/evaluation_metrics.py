import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, auc
from typing import List, Tuple
import matplotlib.pyplot as plt

def calculate_top_k_accuracy(predictions: List[int], ground_truth: List[int], 
                            k: int = 1) -> float:
    """
    Calculate top-K accuracy.
    
    Args:
        predictions: List of predicted identity IDs (sorted by confidence)
        ground_truth: List of true identity IDs
        k: Top K to consider
    
    Returns:
        Top-K accuracy (0-1)
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")
    
    correct = 0
    for pred, truth in zip(predictions, ground_truth):
        if isinstance(pred, list):
            if truth in pred[:k]:
                correct += 1
        else:
            if pred == truth:
                correct += 1
    
    return correct / len(predictions)

def calculate_confusion_matrix(y_true: List[int], y_pred: List[int], 
                              num_classes: int) -> np.ndarray:
    """
    Calculate confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes
    
    Returns:
        Confusion matrix
    """
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    for true, pred in zip(y_true, y_pred):
        matrix[true][pred] += 1
    
    return matrix

def calculate_precision_recall(similarities: List[float], 
                              labels: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate precision-recall curve.
    
    Args:
        similarities: Similarity scores
        labels: Binary labels (1 = match, 0 = non-match)
    
    Returns:
        Tuple of (precision, recall, thresholds)
    """
    precision, recall, thresholds = precision_recall_curve(labels, similarities)
    return precision, recall, thresholds

def calculate_roc_auc(similarities: List[float], 
                     labels: List[int]) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Calculate ROC-AUC.
    
    Args:
        similarities: Similarity scores
        labels: Binary labels
    
    Returns:
        Tuple of (auc_score, fpr, tpr)
    """
    fpr, tpr, _ = roc_curve(labels, similarities)
    roc_auc = auc(fpr, tpr)
    return roc_auc, fpr, tpr

def plot_precision_recall(precision: np.ndarray, recall: np.ndarray, 
                         save_path: str = None):
    """
    Plot precision-recall curve.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='o')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, 
                  roc_auc: float, save_path: str = None):
    """
    Plot ROC curve.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})', marker='o')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()