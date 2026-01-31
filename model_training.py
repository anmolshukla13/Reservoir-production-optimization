"""
Machine Learning Model Training - IMPROVED VERSION
===================================================
Train multiple ML models for reservoir production prediction and optimization.
Includes better error handling, progress tracking, and optional tuning.
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              AdaBoostRegressor, ExtraTreesRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb

# Model Evaluation
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             mean_absolute_percentage_error)
from sklearn.model_selection import cross_val_score, KFold

# Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


class ReservoirMLModels:
    """Train and evaluate ML models for production prediction"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def initialize_models(self):
        """Initialize all models with default parameters"""
        
        self.models = {
            # Linear Models
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(random_state=self.random_state),
            'Lasso': Lasso(random_state=self.random_state),
            'ElasticNet': ElasticNet(random_state=self.random_state),
            
            # Tree-based Models
            'Decision Tree': DecisionTreeRegressor(random_state=self.random_state),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Extra Trees': ExtraTreesRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            # Gradient Boosting
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                random_state=self.random_state
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            ),
            
            # Other Models
            'KNN': KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
        }
        
        print(f"✓ Initialized {len(self.models)} models")
        return self.models
    
    def train_single_model(self, model, X_train, y_train, X_test, y_test, model_name):
        """Train and evaluate a single model"""
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Training {model_name}...", end=" ")
        start_time = datetime.now()
        
        try:
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_metrics = self.calculate_metrics(y_train, y_train_pred)
            test_metrics = self.calculate_metrics(y_test, y_test_pred)
            
            # Training time
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Store results
            results = {
                'model': model,
                'model_name': model_name,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'training_time': training_time,
                'predictions': {
                    'train': y_train_pred,
                    'test': y_test_pred
                }
            }
            
            print(f"✓ R²={test_metrics['r2']:.4f}, RMSE={test_metrics['rmse']:.2f}, Time={training_time:.2f}s")
            
            return results
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            return None
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """Train all initialized models"""
        
        print("\n" + "="*70)
        print("TRAINING ALL MODELS")
        print("="*70)
        
        if not self.models:
            self.initialize_models()
        
        successful_models = 0
        failed_models = []
        
        for name, model in self.models.items():
            results = self.train_single_model(
                model, X_train, y_train, X_test, y_test, name
            )
            if results:
                self.results[name] = results
                successful_models += 1
            else:
                failed_models.append(name)
        
        print("\n" + "="*70)
        print(f"Training Complete: {successful_models}/{len(self.models)} models successful")
        if failed_models:
            print(f"Failed models: {', '.join(failed_models)}")
        print("="*70)
        
        # Find best model
        if self.results:
            self.find_best_model()
        
        return self.results
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate regression metrics"""
        
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100
        }
        
        return metrics
    
    def cross_validate_model(self, model, X, y, cv=5):
        """Perform cross-validation"""
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Running {cv}-fold cross-validation...", end=" ")
        
        kfold = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        scores = {
            'r2': cross_val_score(model, X, y, cv=kfold, 
                                 scoring='r2', n_jobs=-1),
            'neg_mse': cross_val_score(model, X, y, cv=kfold,
                                      scoring='neg_mean_squared_error', n_jobs=-1),
            'neg_mae': cross_val_score(model, X, y, cv=kfold,
                                      scoring='neg_mean_absolute_error', n_jobs=-1)
        }
        
        cv_results = {
            'r2_mean': scores['r2'].mean(),
            'r2_std': scores['r2'].std(),
            'rmse_mean': np.sqrt(-scores['neg_mse'].mean()),
            'mae_mean': -scores['neg_mae'].mean()
        }
        
        print("✓ Complete")
        
        return cv_results
    
    def find_best_model(self, metric='r2'):
        """Find the best performing model"""
        
        if not self.results:
            print("⚠ No models trained yet!")
            return None
        
        best_score = -np.inf
        
        for name, result in self.results.items():
            score = result['test_metrics'][metric]
            if score > best_score:
                best_score = score
                self.best_model = result['model']
                self.best_model_name = name
        
        print(f"\n🏆 Best Model: {self.best_model_name} (Test {metric.upper()}: {best_score:.4f})")
        
        return self.best_model_name, self.best_model
    
    def tune_hyperparameters(self, model_name, X_train, y_train, method='random', n_iter=10):
        """
        Hyperparameter tuning for specific model
        
        Args:
            model_name: Name of model to tune
            X_train, y_train: Training data
            method: 'grid' or 'random'
            n_iter: Number of iterations for random search
        """
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Tuning {model_name} ({method} search)...")
        
        # Define parameter grids (smaller for faster execution)
        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
            },
            'XGBoost': {
                'n_estimators': [100, 200],
                'max_depth': [5, 7, 9],
                'learning_rate': [0.05, 0.1],
                'subsample': [0.8, 1.0],
            },
            'LightGBM': {
                'n_estimators': [100, 200],
                'max_depth': [5, 10, -1],
                'learning_rate': [0.05, 0.1],
                'num_leaves': [31, 50],
            }
        }
        
        if model_name not in param_grids:
            print(f"⚠ No parameter grid defined for {model_name}")
            return None, None
        
        model = self.models[model_name]
        param_grid = param_grids[model_name]
        
        try:
            if method == 'grid':
                search = GridSearchCV(
                    model, param_grid, cv=3, scoring='r2',
                    n_jobs=-1, verbose=0
                )
            else:
                search = RandomizedSearchCV(
                    model, param_grid, n_iter=n_iter, cv=3,
                    scoring='r2', n_jobs=-1, verbose=0,
                    random_state=self.random_state
                )
            
            search.fit(X_train, y_train)
            
            print(f"✓ Best parameters: {search.best_params_}")
            print(f"✓ Best CV score: {search.best_score_:.4f}")
            
            # Update model with best parameters
            self.models[model_name] = search.best_estimator_
            
            return search.best_estimator_, search.best_params_
            
        except Exception as e:
            print(f"✗ Tuning failed: {str(e)}")
            return None, None
    
    def save_model(self, model_name=None, filepath='models/'):
        """Save trained model"""
        
        os.makedirs(filepath, exist_ok=True)
        
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.results:
            print(f"⚠ Model {model_name} not found!")
            return None, None
        
        model = self.results[model_name]['model']
        
        # Save model
        model_file = f"{filepath}{model_name.replace(' ', '_').lower()}_model.pkl"
        joblib.dump(model, model_file)
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'train_metrics': self.results[model_name]['train_metrics'],
            'test_metrics': self.results[model_name]['test_metrics'],
            'training_time': self.results[model_name]['training_time'],
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_file = f"{filepath}{model_name.replace(' ', '_').lower()}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"\n✓ Model saved: {model_file}")
        print(f"✓ Metadata saved: {metadata_file}")
        
        return model_file, metadata_file
    
    def generate_results_report(self):
        """Generate comprehensive results report"""
        
        if not self.results:
            print("⚠ No results available!")
            return None
        
        report = []
        
        for name, result in self.results.items():
            report.append({
                'Model': name,
                'Train_R2': result['train_metrics']['r2'],
                'Test_R2': result['test_metrics']['r2'],
                'Train_RMSE': result['train_metrics']['rmse'],
                'Test_RMSE': result['test_metrics']['rmse'],
                'Train_MAE': result['train_metrics']['mae'],
                'Test_MAE': result['test_metrics']['mae'],
                'MAPE_%': result['test_metrics']['mape'],
                'Training_Time_s': result['training_time']
            })
        
        report_df = pd.DataFrame(report)
        report_df = report_df.sort_values('Test_R2', ascending=False)
        
        return report_df


def main(skip_tuning=False, quick_mode=False):
    """
    Main training pipeline
    
    Args:
        skip_tuning: Skip hyperparameter tuning (faster)
        quick_mode: Use smaller models for quick testing
    """
    
    print("\n" + "="*70)
    print("RESERVOIR PRODUCTION OPTIMIZATION - MODEL TRAINING")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Quick mode: {quick_mode}")
    print(f"Skip tuning: {skip_tuning}")
    print("="*70)
    
    # Load preprocessed data
    print("\n[1/6] Loading and preprocessing data...")
    from preprocessing import preprocess_pipeline
    
    X_train, X_test, y_train, y_test, preprocessor = preprocess_pipeline(
        'data/synthetic/full_dataset.csv',
        target='oil_rate'
    )
    
    # Initialize trainer
    print("\n[2/6] Initializing models...")
    trainer = ReservoirMLModels(random_state=42)
    
    # Modify models for quick mode
    if quick_mode:
        print("⚡ Quick mode: Using smaller models")
        trainer.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=50, n_jobs=-1, random_state=42),
        }
    
    # Train all models
    print("\n[3/6] Training models...")
    results = trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    if not results:
        print("\n✗ No models were successfully trained!")
        return
    
    # Generate report
    print("\n[4/6] Generating comparison report...")
    report = trainer.generate_results_report()
    print("\n" + "="*70)
    print("MODEL COMPARISON REPORT")
    print("="*70)
    print(report.to_string(index=False))
    print("="*70)
    
    # Save report
    os.makedirs('models', exist_ok=True)
    report.to_csv('models/model_comparison_report.csv', index=False)
    print("\n✓ Report saved to models/model_comparison_report.csv")
    
    # Hyperparameter tuning (optional)
    if not skip_tuning and trainer.best_model_name:
        print("\n[5/6] Hyperparameter tuning...")
        best_model, best_params = trainer.tune_hyperparameters(
            trainer.best_model_name,
            X_train, y_train,
            method='random',
            n_iter=5 if quick_mode else 10
        )
    else:
        print("\n[5/6] Skipping hyperparameter tuning")
        best_model = trainer.best_model
    
    # Cross-validation
    if best_model is not None:
        print("\n[6/6] Cross-validation...")
        cv_results = trainer.cross_validate_model(best_model, X_train, y_train, cv=5)
        print("\n" + "="*70)
        print("CROSS-VALIDATION RESULTS")
        print("="*70)
        print(f"R² Score: {cv_results['r2_mean']:.4f} (+/- {cv_results['r2_std']:.4f})")
        print(f"RMSE:     {cv_results['rmse_mean']:.2f}")
        print(f"MAE:      {cv_results['mae_mean']:.2f}")
        print("="*70)
    
    # Save best model
    print("\n[Final] Saving best model...")
    trainer.save_model()
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"✓ Best model: {trainer.best_model_name}")
    print(f"✓ Test R² score: {trainer.results[trainer.best_model_name]['test_metrics']['r2']:.4f}")
    print(f"✓ Models saved in: models/")
    print(f"✓ Report saved: models/model_comparison_report.csv")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    return trainer


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    skip_tuning = '--skip-tuning' in sys.argv
    quick_mode = '--quick' in sys.argv
    
    # Run main training
    trainer = main(skip_tuning=skip_tuning, quick_mode=quick_mode)
    
    print("\n💡 Usage tips:")
    print("  - Quick test:    python model_training_improved.py --quick")
    print("  - Skip tuning:   python model_training_improved.py --skip-tuning")
    print("  - Full training: python model_training_improved.py")