"""
Data transformation module for fraud detection.
Handles normalization, encoding, and imbalanced data treatment.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from typing import Tuple, List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class FraudDataTransformer:
    """
    Handles data transformation including scaling, encoding, and imbalance handling.
    """
    
    def __init__(self, random_state: int = 42, verbose: bool = True):
        """
        Initialize data transformer.
        
        Args:
            random_state (int): Random seed for reproducibility
            verbose (bool): Whether to print transformation steps
        """
        self.random_state = random_state
        self.verbose = verbose
        
        # Transformers
        self.scaler = None
        self.label_encoders = {}
        self.onehot_columns = []
        self.transformation_report = {}
        
    def split_data(self, df: pd.DataFrame, 
                   target_col: str = 'class',
                   test_size: float = 0.2,
                   stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                   pd.Series, pd.Series]:
        """
        Perform stratified train-test split.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Target column name
            test_size (float): Test set proportion
            stratify (bool): Whether to stratify by target
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        if self.verbose:
            print("\n" + "="*60)
            print("STRATIFIED TRAIN-TEST SPLIT")
            print("="*60)
            print("Justification: Preserve class distribution in both sets")
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Calculate class distribution
        class_distribution = y.value_counts(normalize=True)
        
        if self.verbose:
            print(f"Total samples: {len(df):,}")
            print(f"Features: {X.shape[1]}")
            print(f"Class distribution:")
            for cls, pct in class_distribution.items():
                print(f"  Class {cls}: {pct*100:.2f}% ({y.value_counts()[cls]:,} samples)")
        
        # Perform split
        if stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                stratify=y,
                random_state=self.random_state
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=self.random_state
            )
        
        if self.verbose:
            print(f"\n✓ Split completed:")
            print(f"  Training set: {X_train.shape[0]:,} samples")
            print(f"  Test set: {X_test.shape[0]:,} samples")
            print(f"  Train fraud rate: {y_train.mean()*100:.4f}%")
            print(f"  Test fraud rate: {y_test.mean()*100:.4f}%")
            
            # Verify stratification
            if stratify:
                train_dist = y_train.value_counts(normalize=True)
                test_dist = y_test.value_counts(normalize=True)
                
                print(f"\n✅ Stratification verified:")
                print(f"  Original fraud rate: {y.mean()*100:.4f}%")
                print(f"  Train fraud rate: {train_dist[1]*100:.4f}%")
                print(f"  Test fraud rate: {test_dist[1]*100:.4f}%")
        
        self.transformation_report['data_split'] = {
            'train_samples': X_train.shape[0],
            'test_samples': X_test.shape[0],
            'train_fraud_rate': y_train.mean(),
            'test_fraud_rate': y_test.mean(),
            'features_count': X.shape[1]
        }
        
        return X_train, X_test, y_train, y_test
    
    def normalize_numerical_features(self, X_train: pd.DataFrame, 
                                     X_test: pd.DataFrame,
                                     method: str = 'standard') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Normalize numerical features.
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Test features
            method (str): 'standard' (z-score) or 'minmax' (0-1)
            
        Returns:
            tuple: Normalized X_train, X_test
        """
        if self.verbose:
            print("\n" + "="*60)
            print(f"NORMALIZING NUMERICAL FEATURES ({method.upper()})")
            print("="*60)
            print("Justification: Bring features to similar scale for better model convergence")
        
        X_train_norm = X_train.copy()
        X_test_norm = X_test.copy()
        
        # Identify numerical columns
        numerical_cols = X_train_norm.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numerical_cols:
            if self.verbose:
                print("⚠️  No numerical columns found for normalization")
            return X_train_norm, X_test_norm
        
        if self.verbose:
            print(f"Normalizing {len(numerical_cols)} numerical columns")
        
        # Initialize scaler
        if method == 'standard':
            self.scaler = StandardScaler()
            scaler_name = 'StandardScaler (z-score normalization)'
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
            scaler_name = 'MinMaxScaler (0-1 normalization)'
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Fit on training data and transform both sets
        X_train_norm[numerical_cols] = self.scaler.fit_transform(X_train_norm[numerical_cols])
        X_test_norm[numerical_cols] = self.scaler.transform(X_test_norm[numerical_cols])
        
        if self.verbose:
            print(f"✓ Applied {scaler_name}")
            print(f"  Training statistics after normalization:")
            print(f"    Mean ~ 0, Std ~ 1 for StandardScaler")
            print(f"    Range 0-1 for MinMaxScaler")
            
            # Show example statistics
            sample_col = numerical_cols[0]
            print(f"\n  Example column '{sample_col}':")
            print(f"    Train mean: {X_train_norm[sample_col].mean():.4f}")
            print(f"    Train std: {X_train_norm[sample_col].std():.4f}")
            print(f"    Train min: {X_train_norm[sample_col].min():.4f}")
            print(f"    Train max: {X_train_norm[sample_col].max():.4f}")
        
        self.transformation_report['normalization'] = {
            'method': method,
            'numerical_columns': numerical_cols,
            'scaler_type': scaler_name
        }
        
        return X_train_norm, X_test_norm
    
    def encode_categorical_features(self, X_train: pd.DataFrame, 
                                    X_test: pd.DataFrame,
                                    method: str = 'label',
                                    max_categories: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Encode categorical features.
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Test features
            method (str): 'label' or 'onehot'
            max_categories (int): Max unique values for one-hot encoding
            
        Returns:
            tuple: Encoded X_train, X_test
        """
        if self.verbose:
            print("\n" + "="*60)
            print(f"ENCODING CATEGORICAL FEATURES ({method.upper()})")
            print("="*60)
            print("Justification: Convert categorical data to numerical format for ML algorithms")
        
        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy()
        
        # Identify categorical columns
        categorical_cols = X_train_encoded.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
        
        if not categorical_cols:
            if self.verbose:
                print("✓ No categorical columns found")
            return X_train_encoded, X_test_encoded
        
        if self.verbose:
            print(f"Found {len(categorical_cols)} categorical columns:")
            for col in categorical_cols:
                unique_count = X_train_encoded[col].nunique()
                print(f"  {col}: {unique_count} unique values")
        
        if method == 'label':
            # Label encoding
            if self.verbose:
                print("\n🔤 Applying Label Encoding...")
            
            for col in categorical_cols:
                # Initialize label encoder
                le = LabelEncoder()
                
                # Fit on training data
                X_train_encoded[col] = le.fit_transform(X_train_encoded[col].astype(str))
                
                # Transform test data, handling unseen categories
                X_test_encoded[col] = X_test_encoded[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
                
                # Store encoder
                self.label_encoders[col] = le
            
            if self.verbose:
                print(f"✓ Applied label encoding to {len(categorical_cols)} columns")
        
        elif method == 'onehot':
            # One-hot encoding for low-cardinality columns
            if self.verbose:
                print("\n🎯 Applying One-Hot Encoding (for columns with ≤ {max_categories} categories)...")
            
            # Identify columns suitable for one-hot encoding
            onehot_cols = []
            label_cols = []
            
            for col in categorical_cols:
                unique_count = X_train_encoded[col].nunique()
                if unique_count <= max_categories:
                    onehot_cols.append(col)
                else:
                    label_cols.append(col)
            
            if onehot_cols:
                if self.verbose:
                    print(f"  One-hot encoding {len(onehot_cols)} columns:")
                    for col in onehot_cols:
                        print(f"    {col}: {X_train_encoded[col].nunique()} categories")
                
                # Apply one-hot encoding
                X_train_encoded = pd.get_dummies(X_train_encoded, columns=onehot_cols, 
                                                prefix=onehot_cols, drop_first=True)
                X_test_encoded = pd.get_dummies(X_test_encoded, columns=onehot_cols, 
                                               prefix=onehot_cols, drop_first=True)
                
                # Align columns (test might have missing categories)
                X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, 
                                                       fill_value=0)
                
                self.onehot_columns = onehot_cols
            
            # Apply label encoding to high-cardinality columns
            if label_cols:
                if self.verbose:
                    print(f"\n  Label encoding {len(label_cols)} high-cardinality columns:")
                    for col in label_cols:
                        print(f"    {col}: {X_train_encoded[col].nunique()} categories")
                
                for col in label_cols:
                    le = LabelEncoder()
                    X_train_encoded[col] = le.fit_transform(X_train_encoded[col].astype(str))
                    X_test_encoded[col] = X_test_encoded[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
                    self.label_encoders[col] = le
        
        else:
            raise ValueError(f"Unknown encoding method: {method}")
        
        if self.verbose:
            print(f"\n✅ Encoding completed")
            print(f"  Total features after encoding: {X_train_encoded.shape[1]}")
            print(f"  Feature increase: {X_train_encoded.shape[1] - X_train.shape[1]} columns")
        
        self.transformation_report['encoding'] = {
            'method': method,
            'categorical_columns': categorical_cols,
            'onehot_columns': self.onehot_columns if method == 'onehot' else [],
            'label_encoded_columns': list(self.label_encoders.keys()),
            'total_features_after_encoding': X_train_encoded.shape[1]
        }
        
        return X_train_encoded, X_test_encoded
    
    def handle_class_imbalance(self, X_train: pd.DataFrame, y_train: pd.Series,
                              method: str = 'smote',
                              sampling_strategy: float = 0.5) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handle class imbalance using sampling techniques.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            method (str): 'smote', 'undersample', or 'oversample'
            sampling_strategy (float): Desired ratio of minority to majority class
            
        Returns:
            tuple: Resampled X_train, y_train
        """
        if self.verbose:
            print("\n" + "="*60)
            print(f"HANDLING CLASS IMBALANCE ({method.upper()})")
            print("="*60)
            
            # Document justification
            justifications = {
                'smote': "SMOTE creates synthetic samples, preserving information while balancing classes",
                'undersample': "Undersampling reduces majority class, risk of information loss but faster",
                'oversample': "Oversampling duplicates minority class, simple but can cause overfitting"
            }
            
            print(f"Justification: {justifications.get(method, 'Custom sampling strategy')}")
        
        # Analyze class distribution before
        class_counts_before = y_train.value_counts()
        class_pct_before = y_train.value_counts(normalize=True) * 100
        
        if self.verbose:
            print(f"\n📊 Class distribution BEFORE resampling:")
            for cls in sorted(class_counts_before.index):
                print(f"  Class {cls}: {class_counts_before[cls]:,} samples "
                      f"({class_pct_before[cls]:.2f}%)")
            
            imbalance_ratio = class_counts_before[0] / class_counts_before[1]
            print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        X_resampled, y_resampled = X_train.copy(), y_train.copy()
        
        try:
            if method == 'smote':
                # Apply SMOTE
                smote = SMOTE(
                    sampling_strategy=sampling_strategy,
                    random_state=self.random_state,
                    k_neighbors=5
                )
                X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
                method_name = "SMOTE (Synthetic Minority Over-sampling Technique)"
                
            elif method == 'undersample':
                # Apply random undersampling
                undersampler = RandomUnderSampler(
                    sampling_strategy=sampling_strategy,
                    random_state=self.random_state
                )
                X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
                method_name = "Random Undersampling"
                
            elif method == 'oversample':
                # Apply random oversampling
                from imblearn.over_sampling import RandomOverSampler
                oversampler = RandomOverSampler(
                    sampling_strategy=sampling_strategy,
                    random_state=self.random_state
                )
                X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)
                method_name = "Random Oversampling"
                
            else:
                raise ValueError(f"Unknown imbalance handling method: {method}")
            
            # Analyze class distribution after
            class_counts_after = y_resampled.value_counts()
            class_pct_after = y_resampled.value_counts(normalize=True) * 100
            
            if self.verbose:
                print(f"\n✅ Applied {method_name}")
                print(f"\n📊 Class distribution AFTER resampling:")
                for cls in sorted(class_counts_after.index):
                    print(f"  Class {cls}: {class_counts_after[cls]:,} samples "
                          f"({class_pct_after[cls]:.2f}%)")
                
                print(f"\n📈 Resampling statistics:")
                print(f"  Samples added/removed: {len(X_resampled) - len(X_train):,}")
                print(f"  Final training size: {len(X_resampled):,}")
                print(f"  Minority class increase: "
                      f"{((class_counts_after[1] - class_counts_before[1]) / class_counts_before[1] * 100):.1f}%")
                
                # Visualize class distribution
                import matplotlib.pyplot as plt
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Before resampling
                ax1.bar(['Legitimate', 'Fraud'], class_counts_before.values, 
                       color=['#2ecc71', '#e74c3c'])
                ax1.set_title('Class Distribution - BEFORE', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Count', fontsize=10)
                for i, count in enumerate(class_counts_before.values):
                    ax1.text(i, count + count*0.01, f'{count:,}', 
                            ha='center', va='bottom', fontsize=9)
                
                # After resampling
                ax2.bar(['Legitimate', 'Fraud'], class_counts_after.values,
                       color=['#2ecc71', '#e74c3c'])
                ax2.set_title('Class Distribution - AFTER', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Count', fontsize=10)
                for i, count in enumerate(class_counts_after.values):
                    ax2.text(i, count + count*0.01, f'{count:,}', 
                            ha='center', va='bottom', fontsize=9)
                
                plt.tight_layout()
                plt.show()
            
            # Store transformation report
            self.transformation_report['imbalance_handling'] = {
                'method': method,
                'method_name': method_name,
                'sampling_strategy': sampling_strategy,
                'before_counts': class_counts_before.to_dict(),
                'after_counts': class_counts_after.to_dict(),
                'before_percentages': class_pct_before.to_dict(),
                'after_percentages': class_pct_after.to_dict(),
                'samples_added_removed': len(X_resampled) - len(X_train)
            }
            
            return X_resampled, y_resampled
            
        except Exception as e:
            print(f"✗ Error in imbalance handling: {e}")
            print("⚠️  Returning original data")
            return X_train, y_train
    
    def full_pipeline(self, df: pd.DataFrame,
                      target_col: str = 'class',
                      normalize_method: str = 'standard',
                      encode_method: str = 'label',
                      imbalance_method: str = 'smote',
                      test_size: float = 0.2) -> Dict[str, Any]:
        """
        Run complete data transformation pipeline.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Target column name
            normalize_method (str): Normalization method
            encode_method (str): Encoding method
            imbalance_method (str): Imbalance handling method
            test_size (float): Test set proportion
            
        Returns:
            dict: Transformed data and transformers
        """
        if self.verbose:
            print("\n" + "="*80)
            print("COMPLETE DATA TRANSFORMATION PIPELINE")
            print("="*80)
            print("Step 1: Train-test split (stratified)")
            print("Step 2: Categorical feature encoding")
            print("Step 3: Numerical feature normalization")
            print("Step 4: Class imbalance handling (training set only)")
        
        # Step 1: Split data
        X_train, X_test, y_train, y_test = self.split_data(
            df, target_col=target_col, test_size=test_size
        )
        
        # Step 2: Encode categorical features
        X_train_encoded, X_test_encoded = self.encode_categorical_features(
            X_train, X_test, method=encode_method
        )
        
        # Step 3: Normalize numerical features
        X_train_normalized, X_test_normalized = self.normalize_numerical_features(
            X_train_encoded, X_test_encoded, method=normalize_method
        )
        
        # Step 4: Handle class imbalance (only on training data)
        X_train_balanced, y_train_balanced = self.handle_class_imbalance(
            X_train_normalized, y_train, method=imbalance_method
        )
        
        if self.verbose:
            print("\n" + "="*80)
            print("PIPELINE COMPLETED SUCCESSFULLY")
            print("="*80)
            print(f"✓ Final dataset shapes:")
            print(f"  X_train: {X_train_balanced.shape}")
            print(f"  X_test: {X_test_normalized.shape}")
            print(f"  y_train: {y_train_balanced.shape}")
            print(f"  y_test: {y_test.shape}")
            print(f"\n✓ Features after transformation: {X_train_balanced.shape[1]}")
            print(f"✓ Class balance achieved: {y_train_balanced.mean()*100:.2f}% fraud in training")
        
        return {
            'X_train': X_train_balanced,
            'X_test': X_test_normalized,
            'y_train': y_train_balanced,
            'y_test': y_test,
            'transformers': {
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'onehot_columns': self.onehot_columns
            },
            'report': self.transformation_report
        }
    
    def get_transformation_report(self) -> Dict[str, Any]:
        """
        Get comprehensive transformation report.
        
        Returns:
            dict: Transformation statistics and metrics
        """
        return self.transformation_report