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
        """
        Split data into train and test based on seasons.
        
        Args:
            test_seasons: List of seasons to use for testing (e.g., ['2022/2023'])
            
        Returns:
            Tuple of (training_data, test_data)
        """
        mask = self.df['Season'].isin(test_seasons)
        return self.df[~mask], self.df[mask]
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict:
        """Evaluate model performance."""
        results = {
            'accuracy': accuracy_score(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        # Add prediction confidence metrics
        results['avg_confidence'] = np.mean(np.max(y_prob, axis=1))
        results['correct_confidence'] = np.mean([
            prob[true] for prob, true in zip(y_prob, y_true)
        ])
        
        return results
    
    def plot_confusion_matrix(self, cm: np.ndarray):
        """Plot confusion matrix with clear labels."""
        plt.figure(figsize=(10, 8))
        labels = ['Away Win', 'Draw', 'Home Win']
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels
        )
        plt.title('Prediction Results')
        plt.ylabel('Actual Result')
        plt.xlabel('Predicted Result')
        return plt.gcf()
    
    def plot_prediction_distribution(self, y_prob: np.ndarray):
        """Plot distribution of prediction probabilities."""
        plt.figure(figsize=(10, 6))
        labels = ['Away Win', 'Draw', 'Home Win']
        for i, outcome in enumerate(labels):
            plt.hist(y_prob[:, i], alpha=0.5, label=outcome, bins=20)
        plt.title('Distribution of Prediction Probabilities')
        plt.xlabel('Probability')
        plt.ylabel('Count')
        plt.legend()
        return plt.gcf() 