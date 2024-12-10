import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def split_seasons(self, test_seasons: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test based on seasons."""
        mask = self.df['Season'].isin(test_seasons)
        return self.df[~mask], self.df[mask]
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict:
        """Evaluate model performance."""
        results = {
            'accuracy': accuracy_score(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        # Calculate additional metrics
        results['avg_prob_correct'] = np.mean([prob[true] for prob, true in zip(y_prob, y_true)])
        
        return results
    
    def plot_confusion_matrix(self, cm: np.ndarray, title: str = 'Confusion Matrix'):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        return plt.gcf()
    
    def plot_prediction_distribution(self, y_prob: np.ndarray, y_true: np.ndarray):
        """Plot distribution of prediction probabilities."""
        plt.figure(figsize=(10, 6))
        for i, outcome in enumerate(['Away Win', 'Draw', 'Home Win']):
            plt.hist(y_prob[:, i], alpha=0.5, label=outcome, bins=20)
        plt.title('Distribution of Prediction Probabilities')
        plt.xlabel('Probability')
        plt.ylabel('Count')
        plt.legend()
        return plt.gcf() 