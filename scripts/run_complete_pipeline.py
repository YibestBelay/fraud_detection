#!/usr/bin/env python3
"""
Complete pipeline for fraud detection project.
Runs Task 1 (Data Engineering) and Task 3 (Explainability) in sequence.
"""

import sys
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """Run complete fraud detection pipeline."""
    parser = argparse.ArgumentParser(description='Fraud Detection Complete Pipeline')
    parser.add_argument('--task', type=str, default='all',
                       choices=['task1', 'task3', 'all'],
                       help='Which task to run')
    parser.add_argument('--data', type=str, default='fraud_data',
                       choices=['fraud_data', 'creditcard', 'both'],
                       help='Which dataset to use')
    
    args = parser.parse_args()
    
    print("="*80)
    print("FRAUD DETECTION - COMPLETE PIPELINE")
    print("="*80)
    
    if args.task in ['task1', 'all']:
        print("\n🚀 RUNNING TASK 1: DATA ANALYSIS AND PREPROCESSING")
        print("-"*60)
        
        # Task 1 implementation
        from src.data_cleaning import FraudDataCleaner
        from src.eda import FraudEDA
        from src.geolocation import IPGeolocationMapper
        from src.feature_engineering import FraudFeatureEngineer
        from src.data_transformation import FraudDataTransformer
        
        # Run Task 1 steps
        print("✓ Task 1 modules imported")
        print("\nNote: Run notebooks/feature-engineering.ipynb for detailed Task 1 execution")
    
    if args.task in ['task3', 'all']:
        print("\n🚀 RUNNING TASK 3: MODEL EXPLAINABILITY")
        print("-"*60)
        
        # Task 3 implementation
        from src.model_explainability import FraudModelExplainer
        import joblib
        
        print("✓ Task 3 modules imported")
        print("\nNote: Run notebooks/shap-explainability.ipynb for detailed Task 3 execution")
    
    print("\n" + "="*80)
    print("PIPELINE READY")
    print("="*80)
    print("\nTo run complete analysis:")
    print("1. Execute Task 1: jupyter notebook notebooks/feature-engineering.ipynb")
    print("2. Execute Task 2: jupyter notebook notebooks/modeling.ipynb")
    print("3. Execute Task 3: jupyter notebook notebooks/shap-explainability.ipynb")
    print("\nOr run individual scripts as needed.")

if __name__ == "__main__":
    main()