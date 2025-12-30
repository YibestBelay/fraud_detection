
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')


class FraudDataPreprocessor:
    """
    Preprocess fraud detection datasets with handling for imbalanced data.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize preprocessor.
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = None
        
    def load_data(self, filepath, dataset_type='creditcard'):
        """
        Load fraud detection dataset.
        
        Args:
            filepath (str): Path to data file
            dataset_type (str): 'creditcard' or 'fraud_data'
            
        Returns:
            pandas.DataFrame: Loaded dataframe
        """
        try:
            df = pd.read_csv(filepath)
            print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            
            if dataset_type == 'creditcard':
                self.target_column = 'Class'
                # CreditCard dataset already has anonymized features
                if 'Time' in df.columns:
                    df['Hour'] = df['Time'] % 24
                    df = df.drop(['Time'], axis=1)
                    
            elif dataset_type == 'fraud_data':
                self.target_column = 'class'
                # Basic preprocessing for Fraud_Data
                if 'purchase_time' in df.columns:
                    df['purchase_hour'] = pd.to_datetime(df['purchase_time']).dt.hour
                    df = df.drop(['purchase_time'], axis=1)
            
            print(f"Target column: {self.target_column}")
            print(f"Class distribution:\n{df[self.target_column].value_counts()}")
            print(f"Fraud percentage: {df[self.target_column].mean()*100:.2f}%")
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def prepare_features(self, df):
        """
        Separate features and target, handle missing values.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            tuple: (X, y) features and target
        """
        # Identify feature columns
        self.feature_columns = [col for col in df.columns if col != self.target_column]
        
        X = df[self.feature_columns].copy()
        y = df[self.target_column].copy()
        
        # Handle missing values
        if X.isnull().sum().sum() > 0:
            print(f"Missing values found: {X.isnull().sum().sum()}")
            X = X.fillna(X.median())
        
        print(f"Features shape: {X.shape}, Target shape: {y.shape}")
        return X, y
    
    def split_data(self, X, y, test_size=0.2):
        """
        Perform stratified train-test split.
        
        Args:
            X (pandas.DataFrame/numpy.ndarray): Features
            y (pandas.Series/numpy.ndarray): Target
            test_size (float): Test set proportion
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            stratify=y,
            random_state=self.random_state
        )
        
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        print(f"Train fraud rate: {y_train.mean():.4f}")
        print(f"Test fraud rate: {y_test.mean():.4f}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        """
        Scale features using StandardScaler.
        
        Args:
            X_train (pandas.DataFrame/numpy.ndarray): Training features
            X_test (pandas.DataFrame/numpy.ndarray): Test features
            
        Returns:
            tuple: Scaled X_train, X_test
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Features scaled using StandardScaler")
        return X_train_scaled, X_test_scaled
    
    def handle_imbalance(self, X_train, y_train, method='undersample'):
        """
        Handle class imbalance using sampling techniques.
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training target
            method (str): 'undersample', 'oversample', or 'smote'
            
        Returns:
            tuple: Resampled X_train, y_train
        """
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.over_sampling import RandomOverSampler, SMOTE
        
        print(f"\nOriginal class distribution:")
        unique, counts = np.unique(y_train, return_counts=True)
        for cls, cnt in zip(unique, counts):
            print(f"  Class {cls}: {cnt} samples ({cnt/len(y_train)*100:.2f}%)")
        
        if method == 'undersample':
            sampler = RandomUnderSampler(random_state=self.random_state)
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
            print("Applied RandomUnderSampling")
            
        elif method == 'oversample':
            sampler = RandomOverSampler(random_state=self.random_state)
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
            print("Applied RandomOverSampling")
            
        elif method == 'smote':
            sampler = SMOTE(random_state=self.random_state)
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
            print("Applied SMOTE")
            
        else:
            print("No resampling applied")
            return X_train, y_train
        
        print(f"Resampled class distribution:")
        unique, counts = np.unique(y_resampled, return_counts=True)
        for cls, cnt in zip(unique, counts):
            print(f"  Class {cls}: {cnt} samples ({cnt/len(y_resampled)*100:.2f}%)")
        
        return X_resampled, y_resampled