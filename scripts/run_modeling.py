#!/usr/bin/env python3
"""
Script to run the complete fraud detection modeling pipeline.
"""

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import FraudDataPreprocessor
from src.model_training import FraudDetectionModels


def main(dataset_path, dataset_type='creditcard'):
    """
    Run complete modeling pipeline.
    
    Args:
        dataset_path (str): Path to dataset
        dataset_type (str): Type of dataset ('creditcard' or 'fraud_data')
    """
    print("="*80)
    print("FRAUD DETECTION MODELING PIPELINE")
    print("="*80)
    
    try:
        # Step 1: Data Preparation
        print("\n[STEP 1] Data Preparation")
        print("-"*40)
        
        preprocessor = FraudDataPreprocessor(random_state=42)
        df = preprocessor.load_data(dataset_path, dataset_type)
        X, y = preprocessor.prepare_features(df)
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
        X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
        
        # Step 2: Model Training and Evaluation
        print("\n[STEP 2] Model Training and Evaluation")
        print("-"*40)
        
        trainer = FraudDetectionModels(random_state=42)
        
        # Train models
        print("\nTraining models...")
        baseline_model = trainer.train_baseline(X_train_scaled, y_train)
        rf_model = trainer.train_random_forest(X_train_scaled, y_train)
        xgb_model = trainer.train_xgboost(X_train_scaled, y_train)
        
        # Evaluate models
        print("\nEvaluating models...")
        trainer.evaluate_model(baseline_model, X_test_scaled, y_test, 
                              "Logistic Regression")
        trainer.evaluate_model(rf_model, X_test_scaled, y_test, 
                              "Random Forest")
        trainer.evaluate_model(xgb_model, X_test_scaled, y_test, 
                              "XGBoost")
        
        # Cross-validation
        print("\nCross-validation...")
        trainer.cross_validate(baseline_model, X_train_scaled, y_train,
                              model_name="Logistic Regression")
        trainer.cross_validate(rf_model, X_train_scaled, y_train,
                              model_name="Random Forest")
        trainer.cross_validate(xgb_model, X_train_scaled, y_train,
                              model_name="XGBoost")
        
        # Model comparison and selection
        print("\nModel comparison and selection...")
        comparison_df = trainer.compare_models()
        best_name, best_model = trainer.select_best_model(metric='f1')
        
        # Save models
        print("\nSaving models...")
        os.makedirs('../models', exist_ok=True)
        trainer.save_model(baseline_model, '../models/logistic_regression.pkl')
        trainer.save_model(rf_model, '../models/random_forest.pkl')
        trainer.save_model(xgb_model, '../models/xgboost.pkl')
        trainer.save_model(best_model, '../models/best_model.pkl')
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        return {
            'best_model': best_name,
            'comparison': comparison_df,
            'preprocessor': preprocessor,
            'trainer': trainer
        }
        
    except Exception as e:
        print(f"\nError in pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fraud Detection Modeling Pipeline')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to dataset file')
    parser.add_argument('--type', type=str, default='creditcard',
                       choices=['creditcard', 'fraud_data'],
                       help='Type of dataset')
    
    args = parser.parse_args()
    main(args.dataset, args.type)