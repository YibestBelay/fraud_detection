import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')


class FraudDetectionModels:
    """
    Train and evaluate fraud detection models.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize model trainer.
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        
    def train_baseline(self, X_train, y_train, class_weight='balanced'):
        """
        Train baseline Logistic Regression model.
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training target
            class_weight (str/dict): Class weight strategy
            
        Returns:
            sklearn.LogisticRegression: Trained model
        """
        print("=" * 60)
        print("Training Baseline Logistic Regression Model")
        print("=" * 60)
        
        # Initialize model with parameters suitable for imbalanced data
        model = LogisticRegression(
            class_weight=class_weight,
            random_state=self.random_state,
            max_iter=1000,
            solver='liblinear',
            C=1.0,
            penalty='l2'
        )
        
        # Train model
        model.fit(X_train, y_train)
        self.models['logistic_regression'] = model
        
        print("Model trained with parameters:")
        print(f"  Class weight: {class_weight}")
        print(f"  Max iterations: 1000")
        print(f"  Solver: liblinear")
        print(f"  C: 1.0")
        print(f"  Penalty: l2")
        
        return model
    
    def train_random_forest(self, X_train, y_train, n_estimators=100, 
                           max_depth=None, class_weight='balanced'):
        """
        Train Random Forest model.
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training target
            n_estimators (int): Number of trees
            max_depth (int): Maximum tree depth
            class_weight (str/dict): Class weight strategy
            
        Returns:
            sklearn.RandomForestClassifier: Trained model
        """
        print("=" * 60)
        print("Training Random Forest Model")
        print("=" * 60)
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=self.random_state,
            n_jobs=-1,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        
        print("Model trained with parameters:")
        print(f"  n_estimators: {n_estimators}")
        print(f"  max_depth: {max_depth}")
        print(f"  class_weight: {class_weight}")
        print(f"  min_samples_split: 5")
        print(f"  min_samples_leaf: 2")
        
        return model
    
    def train_xgboost(self, X_train, y_train, n_estimators=100, 
                     max_depth=3, learning_rate=0.1):
        """
        Train XGBoost model.
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training target
            n_estimators (int): Number of trees
            max_depth (int): Maximum tree depth
            learning_rate (float): Learning rate
            
        Returns:
            xgboost.XGBClassifier: Trained model
        """
        print("=" * 60)
        print("Training XGBoost Model")
        print("=" * 60)
        
        # Handle class imbalance with scale_pos_weight
        scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train)
        self.models['xgboost'] = model
        
        print("Model trained with parameters:")
        print(f"  n_estimators: {n_estimators}")
        print(f"  max_depth: {max_depth}")
        print(f"  learning_rate: {learning_rate}")
        print(f"  scale_pos_weight: {scale_pos_weight:.2f}")
        
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): Test target
            model_name (str): Name of the model
            
        Returns:
            dict: Evaluation metrics
        """
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}")
        print(f"{'='*60}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'pr_auc': average_precision_score(y_test, y_pred_proba)
        }
        
        # Print metrics
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
        print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")
        
        # Store results
        self.results[model_name] = metrics
        
        return metrics
    
    def cross_validate(self, model, X, y, n_splits=5, model_name="Model"):
        """
        Perform stratified k-fold cross validation.
        
        Args:
            model: Model to cross validate
            X (numpy.ndarray): Features
            y (numpy.ndarray): Target
            n_splits (int): Number of folds
            model_name (str): Name of the model
            
        Returns:
            dict: Cross-validation results
        """
        print(f"\n{'='*60}")
        print(f"Stratified {n_splits}-Fold Cross Validation for {model_name}")
        print(f"{'='*60}")
        
        # Define metrics to calculate
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall', 
            'f1': 'f1',
            'roc_auc': 'roc_auc',
            'average_precision': 'average_precision'
        }
        
        cv_results = {}
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, 
                             random_state=self.random_state)
        
        # Perform cross-validation for each metric
        for metric_name, metric_scorer in scoring.items():
            scores = cross_val_score(
                model, X, y, 
                cv=skf, 
                scoring=metric_scorer,
                n_jobs=-1
            )
            cv_results[metric_name] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'scores': scores
            }
            
            print(f"{metric_name}: {np.mean(scores):.4f} (±{np.std(scores):.4f})")
        
        return cv_results
    
    def compare_models(self):
        """
        Create comparison table of all trained models.
        
        Returns:
            pandas.DataFrame: Comparison table
        """
        if not self.results:
            print("No results to compare. Train and evaluate models first.")
            return None
        
        comparison_df = pd.DataFrame(self.results).T
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        print(comparison_df.round(4))
        
        return comparison_df
    
    def select_best_model(self, metric='f1'):
        """
        Select best model based on specified metric.
        
        Args:
            metric (str): Metric to use for selection (e.g., 'f1', 'pr_auc')
            
        Returns:
            tuple: (best_model_display_name, best_model)
        """
        if not self.results:
            print("No results available for model selection.")
            return None, None

        # Map display names (used in self.results) to internal model keys (used in self.models)
        display_to_internal = {
            "Baseline Logistic Regression": "logistic_regression",
            "Random Forest": "random_forest",
            "XGBoost": "xgboost"
        }

        # Find best model by metric
        metric_values = {name: res[metric] for name, res in self.results.items()}
        best_display_name = max(metric_values, key=metric_values.get)
        
        # Get corresponding internal key
        internal_key = display_to_internal.get(best_display_name)
        if internal_key is None:
            raise KeyError(f"Mapping not found for display name: '{best_display_name}'")
        
        best_model = self.models.get(internal_key)
        if best_model is None:
            raise RuntimeError(f"Model not found for internal key: '{internal_key}'")

        print(f"\n{'='*60}")
        print(f"MODEL SELECTION")
        print(f"{'='*60}")
        print(f"Best model based on {metric}: {best_display_name}")
        print(f"{metric} score: {metric_values[best_display_name]:.4f}")
        print("\nJustification:")
        print("1. For fraud detection, recall and F1-score are critical")
        print("2. Ensemble models typically perform better on imbalanced data")
        print("3. Consider both performance and interpretability")
        
        self.best_model = best_model
        return best_display_name, best_model

    def save_model(self, model, filepath):
        """
        Save trained model to disk.
        
        Args:
            model: Trained model
            filepath (str): Path to save model
        """
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")
    
    def plot_confusion_matrices(self, X_test, y_test):
        """
        Plot confusion matrices for all trained models.
        
        Args:
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): Test target
        """
        n_models = len(self.models)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for ax, (name, model) in zip(axes, self.models.items()):
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'Confusion Matrix - {name}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        plt.tight_layout()
        plt.show()
    
    def plot_precision_recall_curves(self, X_test, y_test):
        """
        Plot precision-recall curves for all trained models.
        
        Args:
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): Test target
        """
        plt.figure(figsize=(10, 8))
        
        for name, model in self.models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = average_precision_score(y_test, y_pred_proba)
            
            plt.plot(recall, precision, lw=2, 
                    label=f'{name} (AP = {pr_auc:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve Comparison')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.show()