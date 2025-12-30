"""
Feature engineering module for fraud detection.
Creates transaction frequency, time-based, and other predictive features.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class FraudFeatureEngineer:
    """
    Feature engineering for fraud detection datasets.
    Creates comprehensive feature set for modeling.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize feature engineer.
        
        Args:
            verbose (bool): Whether to print progress
        """
        self.verbose = verbose
        self.feature_stats = {}
        self.engineered_features = []
        
    def create_time_based_features(self, df: pd.DataFrame,
                                  purchase_time_col: str = 'purchase_time',
                                  signup_time_col: str = 'signup_time') -> pd.DataFrame:
        """
        Create time-based features from transaction timestamps.
        
        Args:
            df (pd.DataFrame): Input dataframe
            purchase_time_col (str): Purchase timestamp column
            signup_time_col (str): Signup timestamp column
            
        Returns:
            pd.DataFrame: Dataframe with time-based features
        """
        df_engineered = df.copy()
        
        if self.verbose:
            print("\n" + "="*60)
            print("CREATING TIME-BASED FEATURES")
            print("="*60)
        
        try:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df_engineered[purchase_time_col]):
                df_engineered[purchase_time_col] = pd.to_datetime(df_engineered[purchase_time_col])
            
            if signup_time_col in df_engineered.columns:
                if not pd.api.types.is_datetime64_any_dtype(df_engineered[signup_time_col]):
                    df_engineered[signup_time_col] = pd.to_datetime(df_engineered[signup_time_col])
            
            # Extract datetime components
            if self.verbose:
                print("📅 Extracting datetime components...")
            
            # Hour of day (0-23)
            df_engineered['purchase_hour'] = df_engineered[purchase_time_col].dt.hour
            
            # Day of week (0=Monday, 6=Sunday)
            df_engineered['purchase_dayofweek'] = df_engineered[purchase_time_col].dt.dayofweek
            
            # Day of month
            df_engineered['purchase_dayofmonth'] = df_engineered[purchase_time_col].dt.day
            
            # Week of year
            df_engineered['purchase_weekofyear'] = df_engineered[purchase_time_col].dt.isocalendar().week
            
            # Month
            df_engineered['purchase_month'] = df_engineered[purchase_time_col].dt.month
            
            # Year
            df_engineered['purchase_year'] = df_engineered[purchase_time_col].dt.year
            
            # Weekend flag
            df_engineered['is_weekend'] = df_engineered['purchase_dayofweek'].isin([5, 6]).astype(int)
            
            # Business hours flag (9 AM - 5 PM)
            df_engineered['is_business_hours'] = ((df_engineered['purchase_hour'] >= 9) & 
                                                 (df_engineered['purchase_hour'] <= 17)).astype(int)
            
            # Time since signup (if signup time available)
            if signup_time_col in df_engineered.columns:
                df_engineered['time_since_signup_days'] = (
                    df_engineered[purchase_time_col] - df_engineered[signup_time_col]
                ).dt.total_seconds() / (24 * 3600)
                
                df_engineered['time_since_signup_hours'] = (
                    df_engineered[purchase_time_col] - df_engineered[signup_time_col]
                ).dt.total_seconds() / 3600
                
                # Flag for new users (transactions within 24 hours of signup)
                df_engineered['is_new_user'] = (df_engineered['time_since_signup_hours'] <= 24).astype(int)
            
            # Time-based cyclical encoding
            if self.verbose:
                print("🔄 Creating cyclical time features...")
            
            # Cyclical encoding for hour (sine/cosine)
            df_engineered['purchase_hour_sin'] = np.sin(2 * np.pi * df_engineered['purchase_hour']/24)
            df_engineered['purchase_hour_cos'] = np.cos(2 * np.pi * df_engineered['purchase_hour']/24)
            
            # Cyclical encoding for day of week
            df_engineered['purchase_day_sin'] = np.sin(2 * np.pi * df_engineered['purchase_dayofweek']/7)
            df_engineered['purchase_day_cos'] = np.cos(2 * np.pi * df_engineered['purchase_dayofweek']/7)
            
            if self.verbose:
                print(f"✓ Created {len([col for col in df_engineered.columns if col not in df.columns])} "
                      f"time-based features")
                
                # Show feature statistics
                time_features = [col for col in df_engineered.columns 
                               if col not in df.columns and 'time' in col.lower() or 'hour' in col.lower()]
                
                print("\n⏰ Time-based features created:")
                for feat in time_features[:10]:  # Show first 10
                    print(f"  - {feat}")
                
                if len(time_features) > 10:
                    print(f"  ... and {len(time_features) - 10} more")
            
            self.engineered_features.extend([col for col in df_engineered.columns 
                                           if col not in df.columns])
            
            return df_engineered
            
        except Exception as e:
            print(f"✗ Error creating time-based features: {e}")
            return df
    
    def create_frequency_features(self, df: pd.DataFrame,
                                 user_id_col: str = 'user_id',
                                 device_id_col: str = 'device_id',
                                 purchase_time_col: str = 'purchase_time',
                                 window_hours: List[int] = [1, 6, 24, 168]) -> pd.DataFrame:
        """
        Create transaction frequency and velocity features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            user_id_col (str): User identifier column
            device_id_col (str): Device identifier column
            purchase_time_col (str): Purchase timestamp column
            window_hours (list): Time windows in hours for frequency calculation
            
        Returns:
            pd.DataFrame: Dataframe with frequency features
        """
        if self.verbose:
            print("\n" + "="*60)
            print("CREATING FREQUENCY & VELOCITY FEATURES")
            print("="*60)
            print("Justification: Fraudsters often make multiple rapid transactions")
        
        df_engineered = df.copy()
        
        try:
            # Ensure datetime type
            if not pd.api.types.is_datetime64_any_dtype(df_engineered[purchase_time_col]):
                df_engineered[purchase_time_col] = pd.to_datetime(df_engineered[purchase_time_col])
            
            # Sort by user and time for rolling calculations
            df_sorted = df_engineered.sort_values([user_id_col, purchase_time_col])
            
            # User-based frequency features
            if self.verbose:
                print("👤 Creating user-based frequency features...")
            
            # Total transactions per user
            user_counts = df_sorted[user_id_col].value_counts()
            df_engineered['user_total_transactions'] = df_engineered[user_id_col].map(user_counts)
            
            # Device-based frequency features
            if device_id_col in df_engineered.columns:
                if self.verbose:
                    print("📱 Creating device-based frequency features...")
                
                # Total transactions per device
                device_counts = df_sorted[device_id_col].value_counts()
                df_engineered['device_total_transactions'] = df_engineered[device_id_col].map(device_counts)
                
                # Unique users per device (multi-accounting detection)
                device_user_counts = df_sorted.groupby(device_id_col)[user_id_col].nunique()
                df_engineered['unique_users_per_device'] = df_engineered[device_id_col].map(device_user_counts)
            
            # Time-window based frequency features
            if self.verbose:
                print("⏱️  Creating time-window frequency features...")
            
            for window in window_hours:
                # Create temporary copy for rolling calculations
                df_temp = df_sorted.copy()
                
                # Calculate time differences
                df_temp['time_diff'] = df_temp.groupby(user_id_col)[purchase_time_col].diff()
                df_temp['time_diff_hours'] = df_temp['time_diff'].dt.total_seconds() / 3600
                
                # Rolling count of transactions within window
                window_label = f'{window}h' if window < 24 else f'{window//24}d'
                
                # Transactions in last X hours
                mask = df_temp['time_diff_hours'] <= window
                rolling_counts = df_temp[mask].groupby(user_id_col).cumcount() + 1
                df_temp[f'transactions_last_{window_label}'] = rolling_counts
                
                # Fill NaN with 0 (first transaction for each user)
                df_temp[f'transactions_last_{window_label}'] = df_temp[f'transactions_last_{window_label}'].fillna(0)
                
                # Map back to original dataframe
                mapping = df_temp.set_index([user_id_col, purchase_time_col])[f'transactions_last_{window_label}']
                df_engineered[f'transactions_last_{window_label}'] = df_engineered.set_index(
                    [user_id_col, purchase_time_col]
                ).index.map(mapping)
            
            # Velocity features (transactions per hour)
            if self.verbose:
                print("⚡ Creating transaction velocity features...")
            
            # Time since first transaction
            user_first_transaction = df_sorted.groupby(user_id_col)[purchase_time_col].min()
            df_engineered['hours_since_first_transaction'] = (
                df_engineered[purchase_time_col] - df_engineered[user_id_col].map(user_first_transaction)
            ).dt.total_seconds() / 3600
            
            # Transactions per hour (velocity)
            df_engineered['transactions_per_hour'] = (
                df_engineered['user_total_transactions'] / 
                np.maximum(df_engineered['hours_since_first_transaction'], 1)
            )
            
            # Flag for rapid transactions
            df_engineered['has_rapid_transactions'] = (
                df_engineered['transactions_last_1h'] > 3
            ).astype(int)
            
            if self.verbose:
                freq_features = [col for col in df_engineered.columns 
                               if col not in df.columns and ('transaction' in col.lower() or 
                                                           'frequency' in col.lower())]
                
                print(f"✓ Created {len(freq_features)} frequency/velocity features")
                print("\n📊 Frequency features created:")
                for feat in freq_features[:10]:
                    print(f"  - {feat}")
                
                if len(freq_features) > 10:
                    print(f"  ... and {len(freq_features) - 10} more")
            
            self.engineered_features.extend([col for col in df_engineered.columns 
                                           if col not in df.columns])
            
            return df_engineered
            
        except Exception as e:
            print(f"✗ Error creating frequency features: {e}")
            return df
    
    def create_aggregate_features(self, df: pd.DataFrame,
                                 group_cols: List[str] = ['user_id', 'device_id', 'country'],
                                 value_cols: List[str] = ['purchase_value', 'age'],
                                 functions: List[str] = ['mean', 'std', 'min', 'max', 'sum']) -> pd.DataFrame:
        """
        Create aggregate features grouped by various columns.
        
        Args:
            df (pd.DataFrame): Input dataframe
            group_cols (list): Columns to group by
            value_cols (list): Columns to aggregate
            functions (list): Aggregation functions
            
        Returns:
            pd.DataFrame: Dataframe with aggregate features
        """
        if self.verbose:
            print("\n" + "="*60)
            print("CREATING AGGREGATE FEATURES")
            print("="*60)
            print("Justification: Behavioral patterns emerge from aggregated statistics")
        
        df_engineered = df.copy()
        
        try:
            available_group_cols = [col for col in group_cols if col in df_engineered.columns]
            available_value_cols = [col for col in value_cols if col in df_engineered.columns]
            
            if not available_group_cols or not available_value_cols:
                print("⚠️  No valid group or value columns for aggregation")
                return df_engineered
            
            for group_col in available_group_cols:
                if self.verbose:
                    print(f"\n📊 Aggregating by {group_col}...")
                
                for value_col in available_value_cols:
                    # Calculate aggregates
                    agg_stats = df_engineered.groupby(group_col)[value_col].agg(functions)
                    
                    # Rename columns
                    agg_stats.columns = [f'{value_col}_{func}_by_{group_col}' 
                                       for func in functions]
                    
                    # Merge with original dataframe
                    df_engineered = df_engineered.merge(
                        agg_stats, 
                        left_on=group_col, 
                        right_index=True,
                        how='left'
                    )
                    
                    # Calculate deviation from group mean
                    mean_col = f'{value_col}_mean_by_{group_col}'
                    if mean_col in df_engineered.columns:
                        df_engineered[f'{value_col}_deviation_from_{group_col}_mean'] = (
                            df_engineered[value_col] - df_engineered[mean_col]
                        )
            
            if self.verbose:
                agg_features = [col for col in df_engineered.columns 
                              if col not in df.columns and any(func in col for func in functions)]
                
                print(f"✓ Created {len(agg_features)} aggregate features")
                print("\n📈 Aggregate features created:")
                for feat in agg_features[:10]:
                    print(f"  - {feat}")
                
                if len(agg_features) > 10:
                    print(f"  ... and {len(agg_features) - 10} more")
            
            self.engineered_features.extend([col for col in df_engineered.columns 
                                           if col not in df.columns])
            
            return df_engineered
            
        except Exception as e:
            print(f"✗ Error creating aggregate features: {e}")
            return df
    
    def create_interaction_features(self, df: pd.DataFrame,
                                   feature_pairs: List[Tuple[str, str]] = None) -> pd.DataFrame:
        """
        Create interaction features between important variables.
        
        Args:
            df (pd.DataFrame): Input dataframe
            feature_pairs (list): Pairs of features to interact
            
        Returns:
            pd.DataFrame: Dataframe with interaction features
        """
        if self.verbose:
            print("\n" + "="*60)
            print("CREATING INTERACTION FEATURES")
            print("="*60)
            print("Justification: Combined effects can be more predictive than individual features")
        
        df_engineered = df.copy()
        
        try:
            # Default interaction pairs if not provided
            if feature_pairs is None:
                # Look for numerical columns for interactions
                numerical_cols = df_engineered.select_dtypes(include=[np.number]).columns.tolist()
                
                # Create some common interactions
                feature_pairs = []
                for i in range(len(numerical_cols)):
                    for j in range(i+1, len(numerical_cols)):
                        if i < 5 and j < 10:  # Limit to avoid explosion
                            feature_pairs.append((numerical_cols[i], numerical_cols[j]))
            
            interaction_count = 0
            
            for feat1, feat2 in feature_pairs:
                if feat1 in df_engineered.columns and feat2 in df_engineered.columns:
                    # Multiplication interaction
                    interaction_name = f'{feat1}_x_{feat2}'
                    df_engineered[interaction_name] = df_engineered[feat1] * df_engineered[feat2]
                    
                    # Ratio interaction (avoid division by zero)
                    ratio_name = f'{feat1}_div_{feat2}'
                    df_engineered[ratio_name] = np.where(
                        df_engineered[feat2] != 0,
                        df_engineered[feat1] / df_engineered[feat2],
                        np.nan
                    )
                    
                    interaction_count += 2
            
            if self.verbose:
                interaction_features = [col for col in df_engineered.columns 
                                      if col not in df.columns and ('_x_' in col or '_div_' in col)]
                
                print(f"✓ Created {len(interaction_features)} interaction features")
                print("\n🤝 Interaction features created:")
                for feat in interaction_features[:10]:
                    print(f"  - {feat}")
                
                if len(interaction_features) > 10:
                    print(f"  ... and {len(interaction_features) - 10} more")
            
            self.engineered_features.extend([col for col in df_engineered.columns 
                                           if col not in df.columns])
            
            return df_engineered
            
        except Exception as e:
            print(f"✗ Error creating interaction features: {e}")
            return df
    
    def create_geolocation_features(self, df: pd.DataFrame,
                                   ip_col: str = 'ip_address',
                                   country_col: str = 'country') -> pd.DataFrame:
        """
        Create features based on geolocation data.
        
        Args:
            df (pd.DataFrame): Input dataframe
            ip_col (str): IP address column
            country_col (str): Country column
            
        Returns:
            pd.DataFrame: Dataframe with geolocation features
        """
        if self.verbose:
            print("\n" + "="*60)
            print("CREATING GEOLOCATION FEATURES")
            print("="*60)
            print("Justification: Fraud patterns vary by geography")
        
        df_engineered = df.copy()
        
        try:
            # IP address features
            if ip_col in df_engineered.columns:
                if self.verbose:
                    print("🌐 Creating IP-based features...")
                
                # Extract IP components if in IPv4 format
                def extract_ip_components(ip):
                    if isinstance(ip, str) and '.' in ip:
                        parts = ip.split('.')
                        if len(parts) == 4:
                            return [int(p) for p in parts]
                    return [np.nan, np.nan, np.nan, np.nan]
                
                ip_components = df_engineered[ip_col].apply(extract_ip_components)
                df_engineered[['ip_octet_1', 'ip_octet_2', 'ip_octet_3', 'ip_octet_4']] = (
                    pd.DataFrame(ip_components.tolist(), index=df_engineered.index)
                )
                
                # IP type features
                df_engineered['is_private_ip'] = (
                    (df_engineered['ip_octet_1'] == 10) |
                    ((df_engineered['ip_octet_1'] == 172) & (df_engineered['ip_octet_2'] >= 16) & 
                     (df_engineered['ip_octet_2'] <= 31)) |
                    ((df_engineered['ip_octet_1'] == 192) & (df_engineered['ip_octet_2'] == 168))
                ).astype(int)
            
            # Country-based features
            if country_col in df_engineered.columns:
                if self.verbose:
                    print("🌍 Creating country-based features...")
                
                # Fraud rate by country (if target available)
                if 'class' in df_engineered.columns:
                    country_fraud_rate = df_engineered.groupby(country_col)['class'].mean()
                    df_engineered['country_fraud_rate'] = df_engineered[country_col].map(country_fraud_rate)
                
                # Transaction count by country
                country_transaction_count = df_engineered[country_col].value_counts()
                df_engineered['country_transaction_count'] = df_engineered[country_col].map(
                    country_transaction_count
                )
                
                # Flag for high-risk countries (top 10% by fraud rate)
                if 'country_fraud_rate' in df_engineered.columns:
                    risk_threshold = df_engineered['country_fraud_rate'].quantile(0.9)
                    df_engineered['is_high_risk_country'] = (
                        df_engineered['country_fraud_rate'] > risk_threshold
                    ).astype(int)
            
            if self.verbose:
                geo_features = [col for col in df_engineered.columns 
                              if col not in df.columns and ('ip_' in col or 'country' in col)]
                
                print(f"✓ Created {len(geo_features)} geolocation features")
                print("\n🗺️  Geolocation features created:")
                for feat in geo_features:
                    print(f"  - {feat}")
            
            self.engineered_features.extend([col for col in df_engineered.columns 
                                           if col not in df.columns])
            
            return df_engineered
            
        except Exception as e:
            print(f"✗ Error creating geolocation features: {e}")
            return df
    
    def create_all_features(self, df: pd.DataFrame,
                           target_col: str = 'class') -> pd.DataFrame:
        """
        Create all feature types in optimal order.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Target column name
            
        Returns:
            pd.DataFrame: Fully engineered dataframe
        """
        if self.verbose:
            print("\n" + "="*80)
            print("COMPREHENSIVE FEATURE ENGINEERING PIPELINE")
            print("="*80)
            print("Step 1: Time-based features")
            print("Step 2: Frequency & velocity features")
            print("Step 3: Aggregate features")
            print("Step 4: Interaction features")
            print("Step 5: Geolocation features")
        
        # Start with original dataframe
        df_engineered = df.copy()
        initial_feature_count = len(df_engineered.columns)
        
        # Remove target column temporarily for feature engineering
        target = None
        if target_col in df_engineered.columns:
            target = df_engineered[target_col]
            df_engineered = df_engineered.drop(columns=[target_col])
        
        # Step 1: Time-based features
        df_engineered = self.create_time_based_features(df_engineered)
        
        # Step 2: Frequency features
        df_engineered = self.create_frequency_features(df_engineered)
        
        # Step 3: Aggregate features
        df_engineered = self.create_aggregate_features(df_engineered)
        
        # Step 4: Interaction features
        df_engineered = self.create_interaction_features(df_engineered)
        
        # Step 5: Geolocation features
        df_engineered = self.create_geolocation_features(df_engineered)
        
        # Add target column back
        if target is not None:
            df_engineered[target_col] = target
        
        # Calculate feature statistics
        final_feature_count = len(df_engineered.columns)
        features_added = final_feature_count - initial_feature_count
        
        if self.verbose:
            print("\n" + "="*60)
            print("FEATURE ENGINEERING SUMMARY")
            print("="*60)
            print(f"Initial features: {initial_feature_count}")
            print(f"Engineered features added: {features_added}")
            print(f"Total features: {final_feature_count}")
            print(f"\nTotal engineered features: {len(self.engineered_features)}")
            
            # Show feature categories
            feature_categories = {
                'Time-based': [f for f in self.engineered_features if any(kw in f.lower() 
                              for kw in ['hour', 'day', 'week', 'month', 'time'])],
                'Frequency': [f for f in self.engineered_features if any(kw in f.lower() 
                             for kw in ['transaction', 'frequency', 'velocity', 'count'])],
                'Aggregate': [f for f in self.engineered_features if any(kw in f.lower() 
                            for kw in ['mean', 'std', 'min', 'max', 'sum', 'by_'])],
                'Interaction': [f for f in self.engineered_features if any(kw in f 
                              for kw in ['_x_', '_div_'])],
                'Geolocation': [f for f in self.engineered_features if any(kw in f.lower() 
                               for kw in ['ip_', 'country'])]
            }
            
            print("\n📊 Feature categories:")
            for category, features in feature_categories.items():
                if features:
                    print(f"  {category}: {len(features)} features")
        
        # Store statistics
        self.feature_stats = {
            'initial_features': initial_feature_count,
            'engineered_features': features_added,
            'total_features': final_feature_count,
            'feature_categories': feature_categories
        }
        
        return df_engineered
    
    def get_feature_importance_report(self, df: pd.DataFrame, 
                                     target_col: str = 'class',
                                     top_n: int = 20) -> pd.DataFrame:
        """
        Generate feature importance report using correlation.
        
        Args:
            df (pd.DataFrame): Engineered dataframe
            target_col (str): Target column name
            top_n (int): Number of top features to show
            
        Returns:
            pd.DataFrame: Feature importance report
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        # Calculate correlation with target
        correlations = {}
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numerical_cols:
            if col != target_col:
                corr = df[[col, target_col]].corr().iloc[0, 1]
                correlations[col] = abs(corr)
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': correlations.keys(),
            'correlation_abs': correlations.values(),
            'correlation': [df[[col, target_col]].corr().iloc[0, 1] for col in correlations.keys()]
        }).sort_values('correlation_abs', ascending=False).head(top_n)
        
        # Categorize features
        def categorize_feature(feature):
            if any(kw in feature for kw in ['hour', 'day', 'week', 'month', 'time']):
                return 'Time-based'
            elif any(kw in feature for kw in ['transaction', 'frequency', 'velocity', 'count']):
                return 'Frequency'
            elif any(kw in feature for kw in ['mean', 'std', 'min', 'max', 'sum', 'by_']):
                return 'Aggregate'
            elif any(kw in feature for kw in ['_x_', '_div_']):
                return 'Interaction'
            elif any(kw in feature for kw in ['ip_', 'country']):
                return 'Geolocation'
            else:
                return 'Original'
        
        importance_df['category'] = importance_df['feature'].apply(categorize_feature)
        
        if self.verbose:
            print("\n" + "="*80)
            print(f"TOP {top_n} FEATURES BY CORRELATION WITH TARGET")
            print("="*80)
            print(importance_df.to_string(index=False))
            
            # Plot feature importance
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(12, 8))
            top_features = importance_df.head(10)
            
            colors = {'Time-based': '#3498db', 'Frequency': '#2ecc71', 
                     'Aggregate': '#e74c3c', 'Interaction': '#f39c12',
                     'Geolocation': '#9b59b6', 'Original': '#95a5a6'}
            
            bar_colors = [colors[cat] for cat in top_features['category']]
            
            bars = plt.barh(range(len(top_features)), top_features['correlation_abs'], 
                          color=bar_colors)
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Absolute Correlation with Target', fontsize=12)
            plt.title('Top 10 Features by Correlation', fontsize=14, fontweight='bold')
            
            # Add correlation values
            for i, (bar, corr) in enumerate(zip(bars, top_features['correlation'])):
                plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{corr:.3f}', va='center', fontsize=10)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=colors[cat], label=cat) 
                             for cat in top_features['category'].unique()]
            plt.legend(handles=legend_elements, loc='lower right')
            
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
        
        return importance_df