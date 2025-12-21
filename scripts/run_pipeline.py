import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.data_cleaning import FraudDataCleaner
from src.eda import FraudEDA
from src.geolocation import IPCountryMapper
from src.feature_engineering import FraudFeatureEngineer
from src.data_transformation import DataTransformer
import pickle
import json

def main():
    """Execute complete fraud detection pipeline."""
    
    # ---------------- CONFIGURATION ----------------
    DATA_PATH = r'C:\Users\admin\fraud-detection-week5\data\raw\Fraud_Data.csv'
    IP_PATH = r'C:\Users\admin\fraud-detection-week5\data\raw\IpAddress_to_Country.csv'
    OUTPUT_DIR = 'data/processed'
    MODEL_DIR = 'models'
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("="*80)
    print("FRAUD DETECTION PIPELINE - START")
    print("="*80)
    
    # ---------------- STEP 1: DATA CLEANING ----------------
    print("\nSTEP 1: DATA CLEANING")
    cleaner = FraudDataCleaner(DATA_PATH, IP_PATH)
    fraud_df, ip_df = cleaner.clean_pipeline()
    
    # Save cleaned data
    fraud_df.to_csv(f'{OUTPUT_DIR}/fraud_data_cleaned.csv', index=False)
    ip_df.to_csv(f'{OUTPUT_DIR}/ip_country_cleaned.csv', index=False)
    print(f"Cleaned data saved to {OUTPUT_DIR}/")
    
    # ---------------- STEP 2: GEOLOCATION ----------------
    print("\nSTEP 2: GEOLOCATION INTEGRATION")
    mapper = IPCountryMapper()
    ip_ranges = mapper.prepare_ip_ranges(ip_df)
    fraud_df = mapper.map_ip_to_country(fraud_df, ip_ranges)
    fraud_df.to_csv(f'{OUTPUT_DIR}/fraud_data_with_country.csv', index=False)
    
    # ---------------- STEP 3: EDA ----------------
    print("\nSTEP 3: EXPLORATORY DATA ANALYSIS")
    eda = FraudEDA(fraud_df)
    class_info = eda.comprehensive_eda()
    
    # Save EDA results
    with open(f'{OUTPUT_DIR}/eda_results.json', 'w') as f:
        json.dump(class_info, f, indent=2)
    
    # ---------------- STEP 4: FEATURE ENGINEERING ----------------
    print("\nSTEP 4: FEATURE ENGINEERING")
    engineer = FraudFeatureEngineer(fraud_df)
    fraud_df_features = engineer.feature_engineering_pipeline()
    fraud_df_features.to_csv(f'{OUTPUT_DIR}/fraud_data_engineered.csv', index=False)
    print(f"Engineered features saved to {OUTPUT_DIR}/fraud_data_engineered.csv")
    
    # ---------------- STEP 5: DATA TRANSFORMATION ----------------
    print("\nSTEP 5: DATA TRANSFORMATION & PREPROCESSING")
    transformer = DataTransformer(target_col='class', test_size=0.2, random_state=42)
    
    # Identify feature types
    feature_types = transformer.identify_feature_types(fraud_df_features)
    
    # Split data
    X_train, X_test, y_train, y_test = transformer.split_data(fraud_df_features)
    
    # Preprocess & optionally handle class imbalance (safe for 8GB RAM)
    processed_data = transformer.preprocess_data(
        X_train, X_test, y_train, y_test,
        feature_types,
        balance_strategy=None  # Set 'smote' or 'undersample' if needed
    )
    
    # Save processed datasets
    processed_data['X_train'].to_csv(f'{OUTPUT_DIR}/X_train.csv', index=False)
    processed_data['X_test'].to_csv(f'{OUTPUT_DIR}/X_test.csv', index=False)
    processed_data['y_train'].to_csv(f'{OUTPUT_DIR}/y_train.csv', index=False)
    processed_data['y_test'].to_csv(f'{OUTPUT_DIR}/y_test.csv', index=False)
    processed_data['X_train_balanced'].to_csv(f'{OUTPUT_DIR}/X_train_balanced.csv', index=False)
    processed_data['y_train_balanced'].to_csv(f'{OUTPUT_DIR}/y_train_balanced.csv', index=False)
    
    # Save preprocessing objects
    with open(f'{MODEL_DIR}/preprocessor.pkl', 'wb') as f:
        pickle.dump(processed_data['preprocessor'], f)
    
    with open(f'{MODEL_DIR}/sampler.pkl', 'wb') as f:
        pickle.dump(processed_data['sampler'], f)
    
    with open(f'{MODEL_DIR}/feature_names.json', 'w') as f:
        json.dump(processed_data['feature_names'], f)
    
    print("\nPIPELINE EXECUTION COMPLETE")
    print("="*80)
    print(f"Outputs saved to {OUTPUT_DIR}/ and models/preprocessor.pkl etc.")
    print("\nNext steps: run modeling.ipynb or shap-explainability.ipynb")

if __name__ == "__main__":
    main()

# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import pandas as pd
# import numpy as np
# from src.data_cleaning import FraudDataCleaner
# from src.eda import FraudEDA
# from src.geolocation import IPCountryMapper
# from src.feature_engineering import FraudFeatureEngineer
# from src.data_transformation import DataTransformer
# import pickle
# import json

# def main():
#     """Execute complete pipeline."""
    
#     # Configuration
#     DATA_PATH = r'C:\Users\admin\fraud-detection-week5\data\raw\Fraud_Data.csv'
#     IP_PATH = r'C:\Users\admin\fraud-detection-week5\data\raw\IpAddress_to_Country.csv'
#     OUTPUT_DIR = 'data/processed'
#     MODEL_DIR = 'models'
    
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     os.makedirs(MODEL_DIR, exist_ok=True)
    
#     print("=" * 80)
#     print("FRAUD DETECTION PIPELINE - COMPLETE EXECUTION")
#     print("=" * 80)
    
#     # 1. DATA CLEANING
#     print("\n" + "=" * 80)
#     print("STEP 1: DATA CLEANING")
#     print("=" * 80)
    
#     cleaner = FraudDataCleaner(DATA_PATH, IP_PATH)
#     fraud_df, ip_df = cleaner.clean_pipeline()
    
#     # Save cleaned data
#     fraud_df.to_csv(f'{OUTPUT_DIR}/fraud_data_cleaned.csv', index=False)
#     ip_df.to_csv(f'{OUTPUT_DIR}/ip_country_cleaned.csv', index=False)
#     print(f"\nSaved cleaned data to {OUTPUT_DIR}/")
    
#     # 2. GEOLOCATION INTEGRATION
#     print("\n" + "=" * 80)
#     print("STEP 2: GEOLOCATION INTEGRATION")
#     print("=" * 80)
    
#     mapper = IPCountryMapper()
#     ip_ranges = mapper.prepare_ip_ranges(ip_df)
#     fraud_df = mapper.map_ip_to_country(fraud_df, ip_ranges)
    
#     # Save with geolocation
#     fraud_df.to_csv(f'{OUTPUT_DIR}/fraud_data_with_country.csv', index=False)
    
#     # 3. EXPLORATORY DATA ANALYSIS
#     print("\n" + "=" * 80)
#     print("STEP 3: EXPLORATORY DATA ANALYSIS")
#     print("=" * 80)
    
#     eda = FraudEDA(fraud_df)
#     class_info = eda.comprehensive_eda()
    
#     # Save EDA results
#     with open(f'{OUTPUT_DIR}/eda_results.json', 'w') as f:
#         json.dump(class_info, f, indent=2)
    
#     # 4. FEATURE ENGINEERING
#     print("\n" + "=" * 80)
#     print("STEP 4: FEATURE ENGINEERING")
#     print("=" * 80)
    
#     engineer = FraudFeatureEngineer(fraud_df)
#     fraud_df_features = engineer.feature_engineering_pipeline()
    
#     # Save engineered features
#     fraud_df_features.to_csv(f'{OUTPUT_DIR}/fraud_data_engineered.csv', index=False)
#     print(f"\nSaved engineered data to {OUTPUT_DIR}/fraud_data_engineered.csv")
    
#     # 5. DATA TRANSFORMATION & PREPROCESSING
#     print("\n" + "=" * 80)
#     print("STEP 5: DATA TRANSFORMATION & PREPROCESSING")
#     print("=" * 80)
    
#     transformer = DataTransformer(target_col='class', test_size=0.2, random_state=42)
    
#     # Identify feature types
#     feature_types = transformer.identify_feature_types(fraud_df_features)
    
#     # Split data
#     X_train, X_test, y_train, y_test = transformer.split_data(fraud_df_features)
    
#     # Preprocess and handle imbalance
#     processed_data = transformer.preprocess_data(
#         X_train, X_test, y_train, y_test,
#         feature_types,
#         # balance_strategy='smote'  
#         balance_strategy=None
#     )
    
#     # Save processed data
#     processed_data['X_train'].to_csv(f'{OUTPUT_DIR}/X_train.csv', index=False)
#     processed_data['X_test'].to_csv(f'{OUTPUT_DIR}/X_test.csv', index=False)
#     processed_data['y_train'].to_csv(f'{OUTPUT_DIR}/y_train.csv', index=False)
#     processed_data['y_test'].to_csv(f'{OUTPUT_DIR}/y_test.csv', index=False)
#     processed_data['X_train_balanced'].to_csv(f'{OUTPUT_DIR}/X_train_balanced.csv', index=False)
#     processed_data['y_train_balanced'].to_csv(f'{OUTPUT_DIR}/y_train_balanced.csv', index=False)
    
#     # Save preprocessing objects
#     with open(f'{MODEL_DIR}/preprocessor.pkl', 'wb') as f:
#         pickle.dump(processed_data['preprocessor'], f)
    
#     with open(f'{MODEL_DIR}/sampler.pkl', 'wb') as f:
#         pickle.dump(processed_data['sampler'], f)
    
#     # Save feature names
#     with open(f'{MODEL_DIR}/feature_names.json', 'w') as f:
#         json.dump(processed_data['feature_names'], f)
    
#     print("\n" + "=" * 80)
#     print("PIPELINE EXECUTION COMPLETE")
#     print("=" * 80)
#     print(f"\nOutput files saved to:")
#     print(f"  - {OUTPUT_DIR}/: Processed data files")
#     print(f"  - {MODEL_DIR}/: Saved preprocessing objects")
#     print(f"\nNext steps:")
#     print("  1. Run modeling.ipynb for model training")
#     print("  2. Run shap-explainability.ipynb for model interpretation")
#     print("  3. Check notebooks/ for visualizations and analysis")

# if __name__ == "__main__":
#     main()