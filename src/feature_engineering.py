"""
Feature engineering that creates value, not just columns.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class FraudFeatureEngineer:
    """Create meaningful features for fraud detection."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.user_activity = {}
        
    def create_time_based_features(self) -> pd.DataFrame:
        """Engineer time-based features with domain knowledge."""
        df = self.df.copy()
        
        print("=== CREATING TIME-BASED FEATURES ===")
        
        # 1. Hour of day
        df['hour_of_day'] = df['purchase_time'].dt.hour
        
        # 2. Day of week
        df['day_of_week'] = df['purchase_time'].dt.dayofweek
        
        # 3. Weekend flag
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # 4. Time since signup (in hours)
        df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
        
        # 5. Month and day
        df['purchase_month'] = df['purchase_time'].dt.month
        df['purchase_day'] = df['purchase_time'].dt.day
        
        print(f"Created {len(['hour_of_day', 'day_of_week', 'is_weekend', 
                            'time_since_signup', 'purchase_month', 'purchase_day'])} time features")
        
        # Validate
        print("\nTime feature statistics:")
        print(f"Average time since signup: {df['time_since_signup'].mean():.2f} hours")
        print(f"Std time since signup: {df['time_since_signup'].std():.2f} hours")
        print(f"Min time since signup: {df['time_since_signup'].min():.2f} hours")
        print(f"Max time since signup: {df['time_since_signup'].max():.2f} hours")
        
        return df
    
    def create_frequency_features(self) -> pd.DataFrame:
        """Calculate transaction frequency and velocity."""
        df = self.df.copy()
        
        print("\n=== CREATING FREQUENCY FEATURES ===")
        
        # Sort by user and time for rolling calculations
        df = df.sort_values(['user_id', 'purchase_time'])
        
        # Initialize features
        df['transactions_last_1h'] = 0
        df['transactions_last_24h'] = 0
        df['transactions_last_7d'] = 0
        
        # Calculate rolling transactions per user
        for user_id in df['user_id'].unique():
            user_mask = df['user_id'] == user_id
            user_df = df[user_mask]
            
            for idx, row in user_df.iterrows():
                current_time = row['purchase_time']
                
                # Transactions in last 1 hour
                mask_1h = (user_df['purchase_time'] >= current_time - timedelta(hours=1)) & \
                         (user_df['purchase_time'] < current_time)
                df.loc[idx, 'transactions_last_1h'] = mask_1h.sum()
                
                # Transactions in last 24 hours
                mask_24h = (user_df['purchase_time'] >= current_time - timedelta(hours=24)) & \
                          (user_df['purchase_time'] < current_time)
                df.loc[idx, 'transactions_last_24h'] = mask_24h.sum()
                
                # Transactions in last 7 days
                mask_7d = (user_df['purchase_time'] >= current_time - timedelta(days=7)) & \
                         (user_df['purchase_time'] < current_time)
                df.loc[idx, 'transactions_last_7d'] = mask_7d.sum()
        
        # Calculate transaction velocity (transactions per hour)
        df['velocity_1h'] = df['transactions_last_1h'] / 1
        df['velocity_24h'] = df['transactions_last_24h'] / 24
        
        print(f"Created frequency and velocity features")
        print("\nFrequency feature statistics:")
        print(f"Average transactions last 1h: {df['transactions_last_1h'].mean():.2f}")
        print(f"Average transactions last 24h: {df['transactions_last_24h'].mean():.2f}")
        print(f"Max transactions last 1h: {df['transactions_last_1h'].max():.2f}")
        
        return df
    
    def create_aggregate_features(self) -> pd.DataFrame:
        """Create aggregate user-level features."""
        df = self.df.copy()
        
        print("\n=== CREATING AGGREGATE FEATURES ===")
        
        # User-level aggregates
        user_stats = df.groupby('user_id').agg({
            'purchase_value': ['mean', 'std', 'max', 'min'],
            'age': 'first',
            'source': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown',
            'browser': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'
        }).reset_index()
        
        # Flatten column names
        user_stats.columns = ['user_id', 'avg_purchase', 'std_purchase', 
                             'max_purchase', 'min_purchase', 'user_age',
                             'common_source', 'common_browser']
        
        # Merge back
        df = pd.merge(df, user_stats, on='user_id', how='left')
        
        # Purchase amount relative to user's average
        df['purchase_deviation'] = df['purchase_value'] - df['avg_purchase']
        df['purchase_zscore'] = (df['purchase_value'] - df['avg_purchase']) / df['std_purchase'].replace(0, 1)
        
        print(f"Created {len(['avg_purchase', 'std_purchase', 'max_purchase', 'min_purchase',
                            'purchase_deviation', 'purchase_zscore'])} aggregate features")
        
        return df
    
    def create_interaction_features(self) -> pd.DataFrame:
        """Create interaction features between variables."""
        df = self.df.copy()
        
        print("\n=== CREATING INTERACTION FEATURES ===")
        
        # Device-browser combination
        df['device_browser'] = df['device_id'].astype(str) + '_' + df['browser'].astype(str)
        df['device_browser'] = df['device_browser'].astype('category')
        
        # Source-browser combination
        df['source_browser'] = df['source'].astype(str) + '_' + df['browser'].astype(str)
        df['source_browser'] = df['source_browser'].astype('category')
        
        # Age groups
        df['age_group'] = pd.cut(df['age'], 
                                bins=[0, 18, 25, 35, 50, 65, 100],
                                labels=['0-18', '19-25', '26-35', '36-50', '51-65', '65+'])
        
        # Purchase value groups
        df['purchase_group'] = pd.qcut(df['purchase_value'], q=5, labels=False)
        
        print(f"Created interaction features")
        
        return df
    
    def feature_engineering_pipeline(self) -> pd.DataFrame:
        """Execute complete feature engineering pipeline."""
        print("=== STARTING FEATURE ENGINEERING PIPELINE ===")
        
        # Store original columns
        original_columns = set(self.df.columns)
        
        # 1. Time-based features
        self.df = self.create_time_based_features()
        
        # 2. Frequency features
        self.df = self.create_frequency_features()
        
        # 3. Aggregate features
        self.df = self.create_aggregate_features()
        
        # 4. Interaction features
        self.df = self.create_interaction_features()
        
        # Identify new features
        new_features = set(self.df.columns) - original_columns
        print(f"\n=== FEATURE ENGINEERING COMPLETE ===")
        print(f"Original features: {len(original_columns)}")
        print(f"New features created: {len(new_features)}")
        print(f"Total features: {len(self.df.columns)}")
        print(f"\nNew features: {sorted(new_features)}")
        
        return self.df