"""Module for training and comparing different models."""
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import shap

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from config import PROCESSED_DATA_DIR
from src.models.dataset_builder import prepare_training_data, load_all_features

class ModelTrainer:
    """Handles model training and comparison."""
    
    def __init__(self):
        """Initialize the trainer."""
        self.models_dir = PROCESSED_DATA_DIR / 'models'
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and prepare data
        print("Loading data...")
        df, feature_groups = load_all_features()
        self.data = prepare_training_data(df, feature_groups)
        
        # Initialize models
        self.models = self._initialize_models()
        
    def _initialize_models(self) -> Dict:
        """Initialize all models to compare.
        
        Returns:
            Dictionary of model instances
        """
        from sklearn.ensemble import RandomForestRegressor
        from xgboost import XGBRegressor
        from sklearn.neural_network import MLPRegressor
        
        return {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            ),
            'xgboost': XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'neural_net': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                max_iter=1000,
                random_state=42
            )
        }
        
    def train_and_evaluate(self) -> Dict:
        """Train and evaluate all models.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(self.data['X_train'], self.data['y_train'])
            
            # Make predictions
            y_pred_train = model.predict(self.data['X_train'])
            y_pred_test = model.predict(self.data['X_test'])
            
            # Calculate metrics
            metrics = {
                'train_r2': r2_score(self.data['y_train'], y_pred_train),
                'test_r2': r2_score(self.data['y_test'], y_pred_test),
                'train_rmse': np.sqrt(mean_squared_error(self.data['y_train'], y_pred_train)),
                'test_rmse': np.sqrt(mean_squared_error(self.data['y_test'], y_pred_test)),
                'train_mae': mean_absolute_error(self.data['y_train'], y_pred_train),
                'test_mae': mean_absolute_error(self.data['y_test'], y_pred_test)
            }
            
            results[name] = metrics
            
            print(f"Train R²: {metrics['train_r2']:.3f}")
            print(f"Test R²: {metrics['test_r2']:.3f}")
            print(f"Test RMSE: {metrics['test_rmse']:.3f}")
        
        return results
        
    def analyze_feature_importance(self, model_name: str = 'xgboost'):
        """Analyze feature importance using SHAP.
        
        Args:
            model_name: Name of model to analyze
        """
        model = self.models[model_name]
        
        # Calculate SHAP values
        print(f"\nCalculating SHAP values for {model_name}...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(self.data['X_test'])
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            self.data['X_test'],
            feature_names=self.data['feature_names'],
            show=False
        )
        plt.tight_layout()
        plt.savefig(self.models_dir / 'shap_importance.png')
        plt.close()
        
        # Calculate and save feature importance scores
        importance_df = pd.DataFrame({
            'feature': self.data['feature_names'],
            'importance': np.abs(shap_values).mean(0)
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Group by feature type
        grouped_importance = {}
        for group_name, group_info in self.data['feature_indices'].items():
            group_features = group_info['features']
            group_importance = importance_df[
                importance_df['feature'].isin(group_features)
            ]['importance'].mean()
            grouped_importance[group_name] = float(group_importance)
        
        # Save importance scores
        with open(self.models_dir / 'feature_importance.json', 'w') as f:
            json.dump({
                'feature_importance': importance_df.to_dict('records'),
                'group_importance': grouped_importance
            }, f, indent=2)
        
        print("\nTop 10 Most Important Features:")
        for _, row in importance_df.head(10).iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")
            
        print("\nFeature Group Importance:")
        for group, score in sorted(grouped_importance.items(), key=lambda x: x[1], reverse=True):
            print(f"{group}: {score:.4f}")
            
    def save_models(self):
        """Save trained models and results."""
        import joblib
        
        # Save each model
        for name, model in self.models.items():
            model_path = self.models_dir / f"{name}.joblib"
            joblib.dump(model, model_path)
            print(f"Saved {name} to {model_path}")
            
        # Save scaler
        scaler_path = self.models_dir / "scaler.joblib"
        joblib.dump(self.data['scaler'], scaler_path)
        print(f"Saved scaler to {scaler_path}")

def main():
    """Train models and analyze results."""
    trainer = ModelTrainer()
    
    # Train and evaluate models
    results = trainer.train_and_evaluate()
    
    # Save results
    with open(PROCESSED_DATA_DIR / 'models/model_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Analyze feature importance
    trainer.analyze_feature_importance()
    
    # Save models
    trainer.save_models()

if __name__ == "__main__":
    main()
