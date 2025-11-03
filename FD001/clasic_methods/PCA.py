import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


class PCAAnomalyDetector:
    """PCA-based anomaly detection for RUL prediction"""
    
    def __init__(self, n_components=5, healthy_threshold=125, percentile=95):
        self.n_components = n_components
        self.healthy_threshold = healthy_threshold
        self.percentile = percentile
        self.pca = PCA(n_components=n_components)
        self.threshold = None
        self.feature_cols = None
        
    def fit(self, train_df, feature_cols):
        """Fit PCA on healthy samples only"""
        self.feature_cols = feature_cols
        
        # Train on healthy engines
        train_healthy = train_df[train_df['RUL'] > self.healthy_threshold][feature_cols]
        self.pca.fit(train_healthy)
        
        # Compute reconstruction error on full train
        train_pca = self.pca.transform(train_df[feature_cols])
        train_reconstructed = self.pca.inverse_transform(train_pca)
        reconstruction_errors = np.mean((train_df[feature_cols].values - train_reconstructed)**2, axis=1)
        
        # Set threshold
        self.threshold = np.percentile(reconstruction_errors, self.percentile)
        
        return self
    
    def compute_reconstruction_error(self, X):
        """Compute reconstruction error for samples"""
        X_pca = self.pca.transform(X)
        X_reconstructed = self.pca.inverse_transform(X_pca)
        return np.mean((X.values - X_reconstructed)**2, axis=1)
    
    def predict_rul(self, test_df, max_rul=150):
        """Predict RUL for test engines"""
        # Compute reconstruction errors
        test_errors = self.compute_reconstruction_error(test_df[self.feature_cols])
        
        # Aggregate by engine
        predictions = []
        for unit_id in test_df['unit_id'].unique():
            engine_mask = test_df['unit_id'] == unit_id
            engine_errors = test_errors[engine_mask]  # This is numpy array
            
            # Fix: use index instead of iloc
            last_error = engine_errors[-1]  # Changed
            
            # Map error to RUL
            degradation_factor = min(last_error / self.threshold, 1.0)
            predicted_rul = max_rul * (1 - degradation_factor)
            
            predictions.append({
                'unit_id': unit_id,
                'predicted_rul': predicted_rul,
                'last_error': last_error,
                'mean_error': engine_errors.mean(),
                'max_error': engine_errors.max()
            })
        
        return pd.DataFrame(predictions)
    
    def evaluate(self, predictions, true_rul):
        """Calculate metrics"""
        pred_rul = predictions['predicted_rul'].values
        true = true_rul['RUL'].values
        
        rmse = np.sqrt(mean_squared_error(true, pred_rul))
        mae = mean_absolute_error(true, pred_rul)
        
        # NASA score
        diff = pred_rul - true
        nasa_score = np.sum(np.where(diff < 0, np.exp(-diff/13)-1, np.exp(diff/10)-1))
        
        return {
            'rmse': rmse,
            'mae': mae,
            'nasa_score': nasa_score,
            'mean_error': (pred_rul - true).mean(),
            'std_error': (pred_rul - true).std()
        }
    
    def plot_results(self, predictions, true_rul):
        """Visualize predictions"""
        pred_rul = predictions['predicted_rul'].values
        true = true_rul['RUL'].values
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Predicted vs True
        axes[0].scatter(true, pred_rul, alpha=0.6)
        min_rul = min(true.min(), pred_rul.min())
        max_rul = max(true.max(), pred_rul.max())
        axes[0].plot([min_rul, max_rul], [min_rul, max_rul], 'r--', label='Perfect')
        axes[0].set_xlabel('True RUL')
        axes[0].set_ylabel('Predicted RUL')
        axes[0].set_title('PCA: Predicted vs True RUL')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Error distribution
        errors = pred_rul - true
        axes[1].hist(errors, bins=30, edgecolor='black', alpha=0.7)
        axes[1].axvline(0, color='red', linestyle='--', label='Zero error')
        axes[1].set_xlabel('Prediction Error')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Error Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_explained_variance(self):
        """Return explained variance info"""
        return {
            'variance_ratio': self.pca.explained_variance_ratio_,
            'cumulative_variance': self.pca.explained_variance_ratio_.cumsum()[-1]
        }