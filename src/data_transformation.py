# """
# Data transformation and class imbalance handling.
# Fully aligned with run_pipeline.py
# Optimized for low RAM (8GB, Windows).
# """

# import pandas as pd
# import numpy as np

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.impute import SimpleImputer
# from imblearn.over_sampling import SMOTE


# class DataTransformer:
#     def __init__(self, target_col, test_size=0.2, random_state=42):
#         self.target_col = target_col
#         self.test_size = test_size
#         self.random_state = random_state

#     # --------------------------------------------------
#     # 1. Identify feature types
#     # --------------------------------------------------
#     def identify_feature_types(self, df):
#         numeric_features = df.select_dtypes(
#             include=["int64", "float64"]
#         ).columns.tolist()

#         categorical_features = df.select_dtypes(
#             include=["object", "category"]
#         ).columns.tolist()

#         # Remove target from features
#         if self.target_col in numeric_features:
#             numeric_features.remove(self.target_col)

#         return {
#             "numeric": numeric_features,
#             "categorical": categorical_features
#         }

#     # --------------------------------------------------
#     # 2. Train / test split
#     # --------------------------------------------------
#     def split_data(self, df):
#         X = df.drop(columns=[self.target_col])
#         y = df[self.target_col]

#         return train_test_split(
#             X, y,
#             test_size=self.test_size,
#             stratify=y,
#             random_state=self.random_state
#         )

#     # --------------------------------------------------
#     # 3. Preprocessing + imbalance handling
#     # --------------------------------------------------
#     def preprocess_data(
#         self,
#         X_train,
#         X_test,
#         y_train,
#         y_test,
#         feature_types,
#         balance_strategy=None
#     ):
#         num_features = feature_types["numeric"]
#         cat_features = feature_types["categorical"]

#         # -------- NUMERIC --------
#         num_imputer = SimpleImputer(strategy="median")
#         scaler = StandardScaler()

#         X_train_num = num_imputer.fit_transform(X_train[num_features])
#         X_test_num = num_imputer.transform(X_test[num_features])

#         X_train_num = scaler.fit_transform(X_train_num)
#         X_test_num = scaler.transform(X_test_num)

#         # -------- CATEGORICAL --------
#         cat_imputer = SimpleImputer(strategy="most_frequent")
#         encoder = OneHotEncoder(
#             handle_unknown="ignore",
#             sparse=True  # VERY IMPORTANT for RAM
#         )

#         X_train_cat = cat_imputer.fit_transform(X_train[cat_features])
#         X_test_cat = cat_imputer.transform(X_test[cat_features])

#         X_train_cat = encoder.fit_transform(X_train_cat)
#         X_test_cat = encoder.transform(X_test_cat)

#         # -------- COMBINE --------
#         from scipy.sparse import hstack

#         X_train_final = hstack([X_train_num, X_train_cat])
#         X_test_final = hstack([X_test_num, X_test_cat])

#         # -------- FEATURE NAMES --------
#         cat_feature_names = encoder.get_feature_names_out(cat_features)
#         feature_names = num_features + list(cat_feature_names)

#         # Convert to DataFrame (safe size)
#         X_train_df = pd.DataFrame(
#             X_train_final.toarray(),
#             columns=feature_names
#         )
#         X_test_df = pd.DataFrame(
#             X_test_final.toarray(),
#             columns=feature_names
#         )

#         # -------- CLASS IMBALANCE --------
#         sampler = None
#         X_train_bal, y_train_bal = X_train_df, y_train

#         print("\nClass distribution BEFORE resampling:")
#         print(y_train.value_counts(normalize=True))

#         if balance_strategy == "smote":
#             sampler = SMOTE(random_state=self.random_state)
#             X_train_bal, y_train_bal = sampler.fit_resample(
#                 X_train_df, y_train
#             )

#             print("\nClass distribution AFTER SMOTE:")
#             print(y_train_bal.value_counts(normalize=True))

#         return {
#             "X_train": X_train_df,
#             "X_test": X_test_df,
#             "y_train": y_train.reset_index(drop=True),
#             "y_test": y_test.reset_index(drop=True),
#             "X_train_balanced": X_train_bal,
#             "y_train_balanced": y_train_bal,
#             "preprocessor": {
#                 "num_imputer": num_imputer,
#                 "scaler": scaler,
#                 "cat_imputer": cat_imputer,
#                 "encoder": encoder
#             },
#             "sampler": sampler,
#             "feature_names": feature_names
#         }

"""
Data transformation and preprocessing pipeline.
Production-ready with proper train/test separation.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

class DataTransformer:
    """Transform and preprocess data for modeling."""
    
    def __init__(self, target_col: str = 'class', test_size: float = 0.2, random_state: int = 42):
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.preprocessor = None
        self.feature_names = None
        
    def identify_feature_types(self, df: pd.DataFrame) -> dict:
        """Identify numerical and categorical features."""
        # Exclude target and non-feature columns
        exclude_cols = [self.target_col, 'user_id', 'device_id', 'signup_time', 
                       'purchase_time', 'ip_address']
        
        feature_df = df.drop(columns=[col for col in exclude_cols if col in df.columns])
        
        # Identify feature types
        numerical_features = feature_df.select_dtypes(include=['int64', 'int32', 'int16', 
                                                             'float64', 'float32']).columns.tolist()
        
        categorical_features = feature_df.select_dtypes(include=['category', 'object']).columns.tolist()
        
        print("=== FEATURE IDENTIFICATION ===")
        print(f"Numerical features ({len(numerical_features)}): {numerical_features}")
        print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
        print(f"Total features: {len(numerical_features) + len(categorical_features)}")
        
        return {
            'numerical': numerical_features,
            'categorical': categorical_features,
            'all': numerical_features + categorical_features
        }
    
    def create_preprocessing_pipeline(self, feature_types: dict) -> ColumnTransformer:
        """Create preprocessing pipeline for different feature types."""
        
        # Numerical pipeline
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),  # Robust to outliers
            ('scaler', StandardScaler())  # Standardize to mean=0, std=1
        ])
        
        # Categorical pipeline
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Column transformer
        preprocessor = ColumnTransformer([
            ('num', numerical_pipeline, feature_types['numerical']),
            ('cat', categorical_pipeline, feature_types['categorical'])
        ])
        
        self.preprocessor = preprocessor
        return preprocessor
    
    def split_data(self, df: pd.DataFrame) -> tuple:
        """Split data into train and test sets with stratification."""
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        
        # Store feature names before preprocessing
        self.feature_names = X.columns.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y,  # Maintain class distribution
            shuffle=True
        )
        
        print("=== DATA SPLITTING ===")
        print(f"Training set: {X_train.shape[0]:,} samples")
        print(f"Test set: {X_test.shape[0]:,} samples")
        print(f"Features: {X_train.shape[1]}")
        print(f"\nTraining class distribution:")
        print(f"  Non-fraud: {(y_train == 0).sum():,} ({(y_train == 0).mean()*100:.2f}%)")
        print(f"  Fraud: {(y_train == 1).sum():,} ({(y_train == 1).mean()*100:.2f}%)")
        
        return X_train, X_test, y_train, y_test
    
    def handle_class_imbalance(self, X_train: pd.DataFrame, y_train: pd.Series, 
                             strategy: str = 'smote') -> tuple:
        """Handle class imbalance with SMOTE or undersampling."""
        
        print(f"\n=== HANDLING CLASS IMBALANCE ({strategy.upper()}) ===")
        
        # Document BEFORE resampling
        print("BEFORE resampling:")
        print(f"  Class 0: {(y_train == 0).sum():,} ({(y_train == 0).mean()*100:.2f}%)")
        print(f"  Class 1: {(y_train == 1).sum():,} ({(y_train == 1).mean()*100:.2f}%)")
        print(f"  Imbalance ratio: {(y_train == 0).sum()/(y_train == 1).sum():.1f}:1")
        
        if strategy == 'smote':
            # JUSTIFICATION: SMOTE creates synthetic minority samples
            # Good when we have enough data and want to preserve all majority samples
            sampler = SMOTE(
                sampling_strategy='auto',  # Balance classes to 1:1
                random_state=self.random_state,
                k_neighbors=5
            )
            justification = """
            JUSTIFICATION FOR SMOTE:
            1. Creates synthetic minority samples rather than discarding data
            2. Preserves all majority class information
            3. Helps prevent overfitting to majority class
            4. Good for tree-based models that we'll likely use
            """
            
        elif strategy == 'undersample':
            # JUSTIFICATION: Random undersampling reduces majority class
            # Good when computational efficiency is important
            sampler = RandomUnderSampler(
                sampling_strategy='auto',
                random_state=self.random_state
            )
            justification = """
            JUSTIFICATION FOR UNDERSAMPLING:
            1. Reduces computational cost
            2. Avoids creating synthetic data (more "real")
            3. Good when dataset is very large
            4. Simpler and faster
            """
        
        elif strategy == 'combined':
            # JUSTIFICATION: Combination of over and under sampling
            # Good for severe imbalance
            over = SMOTE(sampling_strategy=0.5, random_state=self.random_state)
            under = RandomUnderSampler(sampling_strategy=0.8, random_state=self.random_state)
            sampler = ImbPipeline([
                ('over', over),
                ('under', under)
            ])
            justification = """
            JUSTIFICATION FOR COMBINED APPROACH:
            1. Reduces severe imbalance without discarding too much data
            2. Creates some synthetic samples while reducing majority
            3. Balanced approach for very skewed datasets
            """
        
        print(justification)
        
        # Apply resampling
        X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
        
        # Document AFTER resampling
        print("\nAFTER resampling:")
        print(f"  Class 0: {(y_train_resampled == 0).sum():,} ({(y_train_resampled == 0).mean()*100:.2f}%)")
        print(f"  Class 1: {(y_train_resampled == 1).sum():,} ({(y_train_resampled == 1).mean()*100:.2f}%)")
        print(f"  Imbalance ratio: {(y_train_resampled == 0).sum()/(y_train_resampled == 1).sum():.1f}:1")
        
        return X_train_resampled, y_train_resampled, sampler
    
    def preprocess_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                       y_train: pd.Series, y_test: pd.Series, 
                       feature_types: dict, balance_strategy: str = 'smote') -> dict:
        """Complete preprocessing pipeline."""
        
        print("=== STARTING DATA PREPROCESSING PIPELINE ===")
        
        # 1. Create preprocessing pipeline
        preprocessor = self.create_preprocessing_pipeline(feature_types)
        
        # 2. Fit and transform training data
        print("\n1. Fitting preprocessing on training data...")
        X_train_processed = preprocessor.fit_transform(X_train)
        
        # 3. Transform test data (using training fit)
        print("2. Transforming test data...")
        X_test_processed = preprocessor.transform(X_test)
        
        # 4. Handle class imbalance (ONLY on training data)
        print("\n3. Handling class imbalance...")
        X_train_balanced, y_train_balanced, sampler = self.handle_class_imbalance(
            pd.DataFrame(X_train_processed, columns=self.get_feature_names(preprocessor)),
            y_train,
            strategy=balance_strategy
        )
        
        # Convert back to DataFrames with feature names
        feature_names_out = self.get_feature_names(preprocessor)
        
        results = {
            'X_train': pd.DataFrame(X_train_processed, columns=feature_names_out, index=X_train.index),
            'X_test': pd.DataFrame(X_test_processed, columns=feature_names_out, index=X_test.index),
            'y_train': y_train,
            'y_test': y_test,
            'X_train_balanced': pd.DataFrame(X_train_balanced, columns=feature_names_out),
            'y_train_balanced': y_train_balanced,
            'preprocessor': preprocessor,
            'sampler': sampler,
            'feature_names': feature_names_out
        }
        
        print("\n=== PREPROCESSING COMPLETE ===")
        print(f"Processed training shape: {results['X_train'].shape}")
        print(f"Processed test shape: {results['X_test'].shape}")
        print(f"Balanced training shape: {results['X_train_balanced'].shape}")
        
        return results
    
    def get_feature_names(self, preprocessor: ColumnTransformer) -> list:
        """Get feature names after preprocessing."""
        feature_names = []
        
        for name, transformer, columns in preprocessor.transformers_:
            if transformer == 'drop':
                continue
                
            if hasattr(transformer, 'get_feature_names_out'):
                # For OneHotEncoder
                if name == 'cat':
                    feature_names.extend(transformer.named_steps['onehot'].get_feature_names_out(columns))
                else:
                    feature_names.extend(columns)
            else:
                feature_names.extend(columns)
        
        return feature_names